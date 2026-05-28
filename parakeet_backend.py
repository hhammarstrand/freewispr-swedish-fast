"""Parakeet TDT backend wrapping NVIDIA NeMo parakeet-tdt-0.6b-v3.

NeMo is an optional dependency — this module imports cleanly even when NeMo
is not installed. Call is_available() before constructing ParakeetBackend.
"""

import gc
import logging
import math
import os
import queue
import tempfile
import threading
import time
from collections.abc import Callable

import numpy as np

log = logging.getLogger("freewispr")

_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
_STREAM_TARGET_RATE = 16000
# Chunked-batch tick. Every TICK_S the worker re-runs full-window inference
# on whatever audio has accumulated since the last successful pass. 0.3 s
# leaves plenty of headroom: at Parakeet's RTF 0.013, a 5 s window costs
# ~65 ms, well under the 300 ms budget.
_STREAM_TICK_S = 0.3


def is_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import nemo.collections.asr  # noqa: F401
        return True
    except Exception:
        return False


class ParakeetBackend:
    def __init__(self, device: str = "cuda") -> None:
        self._model_lock = threading.RLock()
        self.device = device

        t0 = time.monotonic()
        from nemo.collections.asr import models as nemo_asr
        self.model = nemo_asr.ASRModel.from_pretrained(_MODEL_ID)
        self.model = self.model.to(device)
        self.model.eval()
        log.info("Parakeet '%s' laddad OK [%s] pa %.0f ms",
                 _MODEL_ID, device, (time.monotonic() - t0) * 1000)

        self._warmed = False
        threading.Thread(target=self.warmup, name="parakeet-warmup",
                         daemon=True).start()

    def transcribe_raw(self, audio: np.ndarray) -> str:
        from scipy.io import wavfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            wavfile.write(tmp_path, 16000, audio.astype(np.float32))

            with self._model_lock:
                results = self.model.transcribe([tmp_path])

            if not results:
                return ""
            r = results[0]
            # NeMo 2.x returns list[Hypothesis] with a .text attribute;
            # older releases return list[str].
            text = r.text if hasattr(r, "text") else r
            return text.strip()
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def warmup(self) -> None:
        try:
            t0 = time.monotonic()
            silent = np.zeros(16000, dtype=np.float32)
            with self._model_lock:
                model = self.model
                if model is None:
                    return
            self.transcribe_raw(silent)
            self._warmed = True
            log.info("Parakeet-warmup klar pa %.0f ms", (time.monotonic() - t0) * 1000)
        except Exception as e:
            log.debug("Parakeet-warmup misslyckades (ignoreras): %s", e)

    def close(self) -> None:
        try:
            with self._model_lock:
                model = getattr(self, "model", None)
                if model is None:
                    return
                self.model = None  # type: ignore[assignment]
                del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            log.debug("Kunde inte frigora Parakeet-modell rent: %s", e)

    def start_stream(self,
                     on_partial: Callable[[str], None] | None = None,
                     tick_s: float = _STREAM_TICK_S) -> "StreamingSession":
        """Begin a streaming session for the duration of one hotkey-hold.

        Returns a fresh :class:`StreamingSession` whose worker thread runs
        chunked-batch inference (NeMo's batch ``transcribe`` on growing
        windows) every ``tick_s`` seconds. The model is serialised on the
        backend's :py:attr:`_model_lock`, so callers may keep firing
        ``transcribe_raw`` from other threads concurrently — they'll just
        wait their turn.
        """
        return StreamingSession(self, on_partial=on_partial, tick_s=tick_s)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix to a 1-D float32 array. Selects loudest channel for sparse
    multi-channel inputs (USB mics that put the signal on one side) and
    averages otherwise. Mirrors audio.finalize_audio without importing
    that module (parakeet_backend stays stand-alone)."""
    if audio.ndim <= 1 or audio.shape[1] <= 1:
        return audio.ravel().astype(np.float32, copy=False)
    sums = np.einsum("ij,ij->j", audio, audio)
    loudest = int(np.argmax(sums))
    if sums[loudest] > 0 and float(np.min(sums)) <= float(sums[loudest]) * 0.01:
        return audio[:, loudest].astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def _resample_to_target(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample ``audio`` from ``src_rate`` to 16 kHz. Uses soxr when
    available (≪ 1 ms for 50 ms chunks at 48 kHz) and falls back to
    scipy.signal.resample_poly. Identity when rates already match."""
    if src_rate == _STREAM_TARGET_RATE:
        return audio.astype(np.float32, copy=False)
    try:
        import soxr
        return soxr.resample(audio, src_rate, _STREAM_TARGET_RATE,
                             quality="HQ").astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        g = math.gcd(_STREAM_TARGET_RATE, src_rate)
        up = _STREAM_TARGET_RATE // g
        down = src_rate // g
        return resample_poly(audio, up, down).astype(np.float32)


class StreamingSession:
    """One streaming dictation session.

    Lifecycle: ``push_audio()`` is called from the audio thread (cheap —
    just enqueues raw samples). A background worker thread drains the
    queue, downmixes + resamples to 16 kHz mono, accumulates into a
    growing buffer, and every ``tick_s`` seconds re-runs
    ``ParakeetBackend.transcribe_raw()`` on the full window. The latest
    hypothesis is exposed via ``on_partial`` and as ``self.partial``.

    On ``finalize()``, the worker is signalled to do one last pass and
    the resulting canonical text is returned. The expectation is that
    most of the audio is already covered by the latest partial, so
    finalize completes in well under one full batch-inference latency.
    """

    def __init__(self, backend: "ParakeetBackend",
                 on_partial: Callable[[str], None] | None = None,
                 tick_s: float = _STREAM_TICK_S) -> None:
        self._backend = backend
        self._on_partial = on_partial
        self._tick_s = max(0.05, float(tick_s))

        # Producer/consumer queue between the audio thread and the
        # streaming worker. Bounded only by sanity — at 50 ms chunks for
        # 60 s max recording, 1200 entries is the upper bound.
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=4096)
        self._buffer = np.empty(_STREAM_TARGET_RATE * 30, dtype=np.float32)
        self._buffer_n = 0
        self._buffer_lock = threading.Lock()
        self._last_inferred_n = 0
        self._closed = False
        self.partial: str = ""

        self._stop = threading.Event()
        self._final_requested = threading.Event()
        self._final_done = threading.Event()
        self._worker = threading.Thread(
            target=self._loop, name="parakeet-stream", daemon=True,
        )
        self._worker.start()

    # --------------------------------------------------------- producer

    def push_audio(self, audio: np.ndarray, src_rate: int) -> None:
        """Enqueue a raw audio chunk (any rate / channel count).

        Called from the sounddevice audio callback — must return fast.
        We copy ``audio`` because sounddevice reuses its internal buffer
        on the next callback.
        """
        if self._closed or audio is None or audio.size == 0:
            return
        try:
            self._chunk_queue.put_nowait((np.array(audio, copy=True), int(src_rate)))
        except queue.Full:
            # The worker has fallen behind by 1200+ chunks. Dropping
            # samples is preferable to blocking the audio thread.
            log.warning("Streaming-kö full — droppar ljudchunk")

    # --------------------------------------------------------- consumer

    def _drain_queue(self) -> bool:
        added = False
        while True:
            try:
                chunk, src_rate = self._chunk_queue.get_nowait()
            except queue.Empty:
                break
            try:
                mono = _to_mono(chunk)
                resampled = _resample_to_target(mono, src_rate)
            except Exception:
                log.exception("Streaming-resample misslyckades — droppar chunk")
                continue
            n = resampled.shape[0]
            if n == 0:
                continue
            with self._buffer_lock:
                if self._buffer_n + n > self._buffer.shape[0]:
                    new_cap = max(self._buffer.shape[0] * 2, self._buffer_n + n)
                    new_buf = np.empty(new_cap, dtype=np.float32)
                    new_buf[:self._buffer_n] = self._buffer[:self._buffer_n]
                    self._buffer = new_buf
                self._buffer[self._buffer_n:self._buffer_n + n] = resampled
                self._buffer_n += n
            added = True
        return added

    def _snapshot(self) -> np.ndarray:
        with self._buffer_lock:
            if self._buffer_n == 0:
                return np.empty(0, dtype=np.float32)
            return self._buffer[:self._buffer_n].copy()

    def _loop(self) -> None:
        while True:
            self._stop.wait(self._tick_s)
            is_final = self._final_requested.is_set()
            if self._stop.is_set() and not is_final:
                return
            self._drain_queue()
            n = self._buffer_n
            if n > self._last_inferred_n:
                snap = self._snapshot()
                try:
                    text = self._backend.transcribe_raw(snap)
                    self.partial = text
                    self._last_inferred_n = n
                    cb = self._on_partial
                    if cb is not None and not is_final:
                        try:
                            cb(text)
                        except Exception:
                            log.exception("on_partial callback raised")
                except Exception:
                    log.exception("Streaming partial inference failed")
            if is_final:
                self._final_done.set()
                return

    # --------------------------------------------------------- public API

    def finalize(self, timeout: float = 5.0) -> str:
        """Signal end-of-utterance, run one last inference, return text.

        Idempotent — calling more than once returns the cached partial.
        """
        if self._closed:
            return self.partial
        self._closed = True
        self._final_requested.set()
        # Wait for the worker to finish its final pass. If it's mid-tick,
        # this is short; if it was mid-inference, we wait for that to
        # complete and reuse its output rather than starting another.
        self._final_done.wait(timeout=timeout)
        self._stop.set()
        # Give the worker a moment to exit cleanly so close() doesn't
        # race with an in-flight transcribe.
        self._worker.join(timeout=0.5)
        return self.partial

    def close(self) -> None:
        """Tear down the session without producing a final hypothesis."""
        self._closed = True
        self._stop.set()
        self._final_requested.set()
        self._final_done.set()
        self._worker.join(timeout=0.5)
