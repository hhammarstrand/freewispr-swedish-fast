"""Tests for the streaming-inference path (Parakeet only).

The streaming layer overlaps GPU inference with audio capture so paste
latency at key-release drops to the cost of a tail-segment transcription
plus the post-processing pipeline. The Whisper backend keeps the existing
post-release batch flow — these tests verify both that streaming works
end-to-end behind a config flag *and* that turning the flag off (or
running against Whisper) leaves the legacy path untouched.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
#  Fake Parakeet backend                                                       #
# --------------------------------------------------------------------------- #

class FakeParakeetBackend:
    """Minimal stand-in for ParakeetBackend used by StreamingSession.

    Records every ``transcribe_raw`` call (number of samples and the call
    order). Emits a deterministic hypothesis derived from the buffer length
    so a test can tell partials apart from the final.
    """

    def __init__(self, latency_s: float = 0.0,
                 partials: list[str] | None = None,
                 final: str | None = None) -> None:
        self._model_lock = threading.RLock()
        self.calls: list[int] = []
        self._latency = latency_s
        self._partials = list(partials) if partials else []
        self._final = final

    def transcribe_raw(self, audio: np.ndarray) -> str:
        with self._model_lock:
            self.calls.append(int(audio.size))
            if self._latency:
                time.sleep(self._latency)
            # If the test queued explicit partials, return them in order;
            # the *last* call (after finalize is requested) gets _final.
            if self._partials:
                return self._partials.pop(0)
            return f"hyp:{audio.size}"


# --------------------------------------------------------------------------- #
#  StreamingSession                                                            #
# --------------------------------------------------------------------------- #

def _make_session(backend, tick_s: float = 0.02, on_partial=None):
    """Build a StreamingSession with a fast tick for quick tests."""
    parakeet_backend = importlib.import_module("parakeet_backend")
    return parakeet_backend.StreamingSession(
        backend=backend, on_partial=on_partial, tick_s=tick_s,
    )


def test_streaming_session_accumulates_chunks_and_finalizes():
    backend = FakeParakeetBackend()
    sess = _make_session(backend)

    # Push three 16 kHz mono chunks of 0.1 s each.
    chunk = np.ones(1600, dtype=np.float32) * 0.1
    sess.push_audio(chunk, 16000)
    sess.push_audio(chunk, 16000)
    sess.push_audio(chunk, 16000)

    text = sess.finalize(timeout=2.0)
    # The final hypothesis must reflect roughly 3 chunks @ 1600 samples =
    # 4800 (the worker may have run partial passes on a subset earlier).
    assert text.startswith("hyp:")
    last_n = int(text.split(":")[1])
    assert last_n == 4800
    # At least one transcribe call happened, and the LAST call was on the
    # full buffer.
    assert backend.calls
    assert backend.calls[-1] == 4800


def test_streaming_session_resamples_native_rate_to_16k(monkeypatch):
    # An earlier test in this collection swaps sys.modules["torch"] for a
    # minimal SimpleNamespace that lacks ``.Tensor``. scipy.signal.resample_poly
    # probes torch.Tensor via array-api compat and AttributeErrors out, which
    # would mask the actual resample logic we want to exercise here. Restore
    # a stub that has every attribute scipy probes for.
    if "torch" in sys.modules:
        existing = sys.modules["torch"]
        if not hasattr(existing, "Tensor"):
            class _FakeTensor:
                pass
            patched = SimpleNamespace(
                Tensor=_FakeTensor,
                is_tensor=lambda x: False,
                cuda=getattr(existing, "cuda", SimpleNamespace(
                    is_available=lambda: False, empty_cache=lambda: None)),
            )
            monkeypatch.setitem(sys.modules, "torch", patched)

    backend = FakeParakeetBackend()
    sess = _make_session(backend)

    # 0.1 s @ 48 kHz stereo → 4800 samples * 2 ch.
    stereo_48k = np.zeros((4800, 2), dtype=np.float32)
    stereo_48k[:, 0] = 0.5  # active left channel
    sess.push_audio(stereo_48k, 48000)

    sess.finalize(timeout=2.0)
    # After resample to 16 kHz mono we expect ~1600 samples. Resample
    # filters may produce 1599 or 1601 depending on quality setting.
    assert backend.calls
    final_n = backend.calls[-1]
    assert 1500 <= final_n <= 1700, f"expected ~1600 samples, got {final_n}"


def test_streaming_session_fires_on_partial_callback():
    backend = FakeParakeetBackend()
    partials: list[str] = []

    sess = _make_session(backend, on_partial=partials.append)
    chunk = np.ones(800, dtype=np.float32) * 0.1
    for _ in range(4):
        sess.push_audio(chunk, 16000)
        # Let the worker have at least one tick to run.
        time.sleep(0.04)

    sess.finalize(timeout=2.0)
    # The worker fires on_partial during the hold (NOT during finalize),
    # so the list should contain at least one entry from before finalize.
    assert partials, "on_partial callback never fired"
    assert all(p.startswith("hyp:") for p in partials)


def test_streaming_session_finalize_is_idempotent():
    backend = FakeParakeetBackend()
    sess = _make_session(backend)
    sess.push_audio(np.ones(1600, dtype=np.float32), 16000)

    first = sess.finalize(timeout=2.0)
    second = sess.finalize(timeout=2.0)
    assert first == second
    # No additional inference work should happen on the second call.
    n_calls_before_second = len(backend.calls)
    sess.finalize(timeout=0.1)
    assert len(backend.calls) == n_calls_before_second


def test_streaming_session_close_stops_worker_quickly():
    backend = FakeParakeetBackend()
    sess = _make_session(backend, tick_s=0.05)
    sess.push_audio(np.ones(800, dtype=np.float32), 16000)
    t0 = time.perf_counter()
    sess.close()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"close took {elapsed:.2f}s — worker not exiting"
    assert not sess._worker.is_alive()


# --------------------------------------------------------------------------- #
#  StreamingHandle                                                             #
# --------------------------------------------------------------------------- #

def _fresh_transcriber_module():
    """Reload transcriber.py against the conftest stubs.

    The module imports ``faster_whisper`` and ``corrections`` at top level;
    the conftest already substitutes the former on Linux CI. We just reload
    so any prior monkeypatches from other tests don't leak in.
    """
    sys.modules.setdefault("faster_whisper",
                           SimpleNamespace(WhisperModel=object))
    if "transcriber" in sys.modules:
        del sys.modules["transcriber"]
    return importlib.import_module("transcriber")


def test_streaming_handle_applies_postprocess_at_finalize(tmp_path, monkeypatch):
    transcriber_mod = _fresh_transcriber_module()
    # Empty corrections so the post-processor only touches normalization.
    corrections = importlib.import_module("corrections")
    if hasattr(corrections, "_store"):
        corrections._store._path = tmp_path / "corrections.json"
        corrections._store._cache = None
        corrections._store._cache_mtime = 0.0
    corrections.save({})

    # Hand-craft a Transcriber-like object so we don't need to load Whisper.
    t = object.__new__(transcriber_mod.Transcriber)

    class FakeStreamSession:
        def push_audio(self, audio, rate): pass
        def finalize(self, timeout=5.0):
            # Lower-case, with noise marker + missing space — exercises
            # the noise-strip, postprocess capitalization, and spacing.
            return "[BLANK_AUDIO] hej.du"
        def close(self): pass

    handle = transcriber_mod.StreamingHandle(
        FakeStreamSession(), post=t._post_process,
    )
    text = handle.finalize()
    # _postprocess strips the noise placeholder, capitalises the first
    # letter of the whole text, and inserts the missing space after the
    # period. It does not re-capitalise after the period (Whisper output
    # already mid-sentence often legitimately lower-cases the next word),
    # so "Hej. du" is the expected canonical form.
    assert "[BLANK_AUDIO]" not in text
    assert text == "Hej. du"


def test_streaming_handle_finalize_is_idempotent():
    transcriber_mod = _fresh_transcriber_module()
    calls = []

    class FakeStreamSession:
        def push_audio(self, audio, rate): pass
        def finalize(self, timeout=5.0):
            calls.append("finalize")
            return "raw"
        def close(self): pass

    handle = transcriber_mod.StreamingHandle(
        FakeStreamSession(), post=lambda t: t.upper(),
    )
    assert handle.finalize() == "RAW"
    assert handle.finalize() == "RAW"
    assert calls == ["finalize"]


def test_transcriber_start_stream_returns_none_for_whisper_backend():
    """No Parakeet → no streaming. Caller must fall back to batch."""
    transcriber_mod = _fresh_transcriber_module()
    t = object.__new__(transcriber_mod.Transcriber)
    t._parakeet = None
    assert t.start_stream() is None


def test_transcriber_start_stream_wraps_parakeet_session():
    transcriber_mod = _fresh_transcriber_module()
    t = object.__new__(transcriber_mod.Transcriber)

    captured = {}
    sentinel_session = SimpleNamespace(name="parakeet-session")

    class FakeParakeet:
        def start_stream(self, on_partial=None):
            captured["on_partial"] = on_partial
            return sentinel_session

    t._parakeet = FakeParakeet()

    cb = lambda text: None
    handle = t.start_stream(on_partial=cb)
    assert isinstance(handle, transcriber_mod.StreamingHandle)
    assert captured["on_partial"] is cb
    assert handle._sess is sentinel_session


# --------------------------------------------------------------------------- #
#  Dictation integration                                                       #
# --------------------------------------------------------------------------- #

def _fresh_dictation_module(monkeypatch):
    """Reload dictation against minimal stubs for sounds + keyboard."""
    monkeypatch.setitem(sys.modules, "sounds", SimpleNamespace(
        play_start=lambda: None, play_stop=lambda: None, play_error=lambda: None,
    ))
    # Keyboard already stubbed by conftest; ensure parse helpers exist.
    if "dictation" in sys.modules:
        del sys.modules["dictation"]
    return importlib.import_module("dictation")


class _FakeStreamHandle:
    """Drop-in replacement for transcriber.StreamingHandle in tests."""

    def __init__(self, final_text: str = "final text") -> None:
        self.pushed: list[tuple[int, int]] = []
        self.finalized = False
        self.closed = False
        self._final_text = final_text

    def push_audio(self, audio, rate):
        self.pushed.append((audio.size, rate))

    def finalize(self, timeout=5.0):
        self.finalized = True
        return self._final_text

    def close(self):
        self.closed = True


def _make_dictation_mode(dictation_mod, *, transcriber, streaming: bool):
    """Construct a DictationMode without starting the worker / hooks."""
    mode = object.__new__(dictation_mod.DictationMode)
    mode.transcriber = transcriber
    mode.hotkey = "ctrl+space"
    mode.streaming = streaming
    mode._stream = None
    mode._active = True
    mode._recording = False
    mode._hook_handles = []
    mode._modifiers = ()
    mode._modifier_keys = ()
    mode.min_rms = 0.0  # let any RMS through
    mode.paste_strategy = "auto"
    mode.paste_threshold = 200
    mode.on_status = lambda msg: None
    mode.indicator = None
    mode._worker_stop = threading.Event()
    return mode


def test_dictation_streaming_path_uses_session_text(monkeypatch):
    dictation_mod = _fresh_dictation_module(monkeypatch)

    pasted: list[str] = []
    monkeypatch.setattr(
        dictation_mod, "inject_text",
        lambda text, active_modifiers=(), strategy="auto", paste_threshold=200:
            pasted.append(text) or True,
    )

    handle = _FakeStreamHandle(final_text="hej från streaming")
    mode = _make_dictation_mode(
        dictation_mod,
        transcriber=SimpleNamespace(llm_enabled=False, llm_api_key=""),
        streaming=True,
    )
    # Audio sample length must clear MIN_AUDIO_SAMPLES (3200 @ 16 kHz, scaled
    # to native rate of 48 kHz → 9600 samples).
    raw = np.ones(16000, dtype=np.float32) * 0.1

    mode._process_job(raw, channels=1, rate=48000, rms=0.1, stream=handle)

    assert handle.finalized is True
    assert handle.closed is False  # finalize subsumes close
    assert pasted == ["hej från streaming"]


def test_dictation_streaming_closes_session_on_too_short(monkeypatch):
    """Short recordings must abort the session without pasting."""
    dictation_mod = _fresh_dictation_module(monkeypatch)
    monkeypatch.setattr(dictation_mod, "inject_text",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("must not paste on short audio")))

    handle = _FakeStreamHandle()
    mode = _make_dictation_mode(
        dictation_mod,
        transcriber=SimpleNamespace(llm_enabled=False, llm_api_key=""),
        streaming=True,
    )
    # 100 samples @ 16 kHz is well below MIN_AUDIO_SAMPLES = 3200.
    raw = np.ones(100, dtype=np.float32)
    mode._process_job(raw, channels=1, rate=16000, rms=0.1, stream=handle)

    assert handle.closed is True
    assert handle.finalized is False


def test_dictation_streaming_closes_session_on_too_quiet(monkeypatch):
    dictation_mod = _fresh_dictation_module(monkeypatch)
    monkeypatch.setattr(dictation_mod, "inject_text",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("must not paste on silence")))

    handle = _FakeStreamHandle()
    mode = _make_dictation_mode(
        dictation_mod,
        transcriber=SimpleNamespace(llm_enabled=False, llm_api_key=""),
        streaming=True,
    )
    mode.min_rms = 0.01
    raw = np.zeros(16000, dtype=np.float32)
    mode._process_job(raw, channels=1, rate=16000, rms=0.001, stream=handle)

    assert handle.closed is True
    assert handle.finalized is False


def test_dictation_falls_back_to_batch_when_no_stream(monkeypatch):
    """Without a streaming session in the job, the existing batch path runs."""
    dictation_mod = _fresh_dictation_module(monkeypatch)

    pasted: list[str] = []
    monkeypatch.setattr(
        dictation_mod, "inject_text",
        lambda text, active_modifiers=(), strategy="auto", paste_threshold=200:
            pasted.append(text) or True,
    )
    # finalize_audio passes its argument through (mono 16 kHz already).
    monkeypatch.setattr(dictation_mod, "finalize_audio",
                        lambda audio, channels, rate: audio)

    transcriber_calls = []

    class FakeBatchTranscriber:
        llm_enabled = False
        llm_api_key = ""
        def transcribe_local(self, audio):
            transcriber_calls.append(int(audio.size))
            return "batch result"

    mode = _make_dictation_mode(
        dictation_mod, transcriber=FakeBatchTranscriber(), streaming=False,
    )
    raw = np.ones(16000, dtype=np.float32) * 0.1
    mode._process_job(raw, channels=1, rate=16000, rms=0.1, stream=None)

    assert pasted == ["batch result"]
    assert transcriber_calls == [16000]


def test_dictation_streaming_config_default_is_off():
    """Streaming must default to OFF until bench validates real-world WER."""
    config = importlib.import_module("config")
    assert config.DEFAULTS["streaming"] is False


def test_audio_callback_fires_on_chunk_with_native_rate():
    audio = importlib.reload(importlib.import_module("audio"))
    recorder = audio.MicRecorder()
    recorder._ensure_buffer(rate=48000, channels=1)
    recorder._rate = 48000
    recorder.recording = True

    received: list[tuple[int, int]] = []
    recorder.on_chunk = lambda chunk, rate: received.append((chunk.size, rate))

    indata = np.linspace(0.1, 0.2, 480, dtype=np.float32).reshape(-1, 1)
    recorder._cb(indata, len(indata), None, None)

    assert received == [(480, 48000)]


def test_audio_callback_on_chunk_exception_does_not_crash_audio_thread():
    audio = importlib.reload(importlib.import_module("audio"))
    recorder = audio.MicRecorder()
    recorder._ensure_buffer(rate=16000, channels=1)
    recorder._rate = 16000
    recorder.recording = True

    def angry_chunk(_chunk, _rate):
        raise RuntimeError("downstream is broken")
    recorder.on_chunk = angry_chunk

    indata = np.ones(160, dtype=np.float32).reshape(-1, 1)
    # The exception must be swallowed — a raise here would mean the next
    # PortAudio callback never fires.
    recorder._cb(indata, len(indata), None, None)
    # Buffer write still happened.
    assert recorder._buffer_offset == 160


def test_dictation_stop_closes_pending_stream(monkeypatch):
    dictation_mod = _fresh_dictation_module(monkeypatch)
    monkeypatch.setattr(dictation_mod, "keyboard",
                        SimpleNamespace(unhook=lambda h: None,
                                        is_pressed=lambda k: False))

    mode = _make_dictation_mode(
        dictation_mod,
        transcriber=SimpleNamespace(llm_enabled=False, llm_api_key=""),
        streaming=True,
    )
    handle = _FakeStreamHandle()
    mode._stream = handle
    mode.recorder = SimpleNamespace(on_chunk=lambda *a, **kw: None)
    import queue as _queue
    mode._jobs = _queue.Queue(maxsize=2)
    mode._worker_thread = None

    mode.stop(wait=False)
    assert handle.closed is True
    assert mode._stream is None
