import logging
import queue
import threading

import keyboard
import numpy as np

import snippets as snippet_module
import sounds
from audio import MicRecorder, finalize_audio
from modifiers import is_modifier, normalize_all
from text_inject import inject as inject_text
from transcriber import Transcriber

log = logging.getLogger("freewispr")

MIN_AUDIO_SAMPLES = 3200   # 0.2 s at 16 kHz — ignore accidental taps
# Default RMS gate. Audio quieter than this is treated as silence and dropped
# without invoking Whisper (saves ~1 s of CPU per phantom press).
#
# Derivation: with int16 → float32 normalisation the noise floor of a typical
# USB headset in a quiet room measures RMS ≈ 0.0005-0.001. A whispered word
# sits around 0.005-0.01, normal speech 0.02-0.1. 0.003 leaves comfortable
# headroom above silence while still letting through quiet speech.
# Overridable via DictationMode(min_rms=...) — surfaced in Settings as
# "Lägsta inspelningsnivå".
DEFAULT_MIN_RMS = 0.003

# Bounded queue prevents memory blow-up if the user spams the hotkey while
# transcriptions stall (e.g. LLM-polish round-trip). Beyond this depth we
# drop new presses and show "Upptagen" instead.
_QUEUE_MAX = 2


def _text_meta(text: str) -> str:
    words = len(text.split())
    return f"chars={len(text)}, words={words}"


def _parse_hotkey(hotkey: str) -> tuple[str, tuple[str, ...]]:
    """Split a hotkey string into (trigger, canonical_modifiers).

    Falls back to naive ``+`` splitting. Modifier names are normalised via
    :py:mod:`modifiers` so ``cmd``, ``win``, ``windows`` all map to the
    same canonical ``windows`` token used by the paste layer.
    """
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    if not parts:
        return hotkey.strip().lower(), ()
    trigger = parts[-1]
    raw_modifiers = parts[:-1]
    modifiers = normalize_all(raw_modifiers)
    # Preserve unknown non-modifier prefixes so the held-check still gates
    # on them (rare; e.g. user types ``foo+bar`` deliberately).
    if not modifiers and len(raw_modifiers) > 0 and not any(is_modifier(m) for m in raw_modifiers):
        modifiers = tuple(raw_modifiers)
    return trigger, modifiers


class DictationMode:
    def __init__(self, transcriber: Transcriber, hotkey: str = "ctrl+space",
                 on_status=None, indicator=None,
                 mic_device: str | dict | None = None,
                 min_rms: float = DEFAULT_MIN_RMS,
                 paste_strategy: str = "auto",
                 paste_threshold: int = 200,
                 streaming: bool = False):
        self.transcriber = transcriber
        self.hotkey = hotkey
        # MicRecorder accepts str (legacy), dict (structured), or None.
        self.recorder = MicRecorder(device=mic_device)
        self.on_status = on_status or (lambda msg: None)
        self.indicator = indicator
        self.min_rms = min_rms
        self.paste_strategy = paste_strategy
        self.paste_threshold = int(paste_threshold)
        # Streaming: run Parakeet inference during the hold for sub-100 ms
        # paste latency. Falls back to the post-release batch path when the
        # backend doesn't support streaming (Whisper) or when explicitly off.
        self.streaming = bool(streaming)
        self._stream = None  # type: ignore[assignment]
        self._active = False
        self._recording = False
        self._hook_handles: list = []

        # Bounded queue + dedicated worker thread. Replaces the old single-slot
        # lock that *dropped* recordings — instead we queue up to _QUEUE_MAX
        # and only drop if the queue is full.
        self._jobs: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker_stop = threading.Event()
        self._worker_thread: threading.Thread | None = None

        self._trigger_key, self._modifiers = _parse_hotkey(hotkey)
        # Cached tuple passed to paste_text — lets the paste layer release
        # only the modifiers from this hotkey (not all of them — releasing
        # an unheld Win key opens the Start menu on Windows).
        self._modifier_keys: tuple[str, ...] = self._modifiers

    # ------------------------------------------------------------------ public

    def start(self):
        self._active = True
        log.info("Hotkey: trigger='%s', modifiers=%s",
                 self._trigger_key, list(self._modifiers))
        # Start the transcription worker before installing hooks so the very
        # first press has a consumer ready.
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="dictation-worker", daemon=True)
        self._worker_thread.start()
        # Track our own handles so stop() can detach cleanly without nuking
        # keyboard hooks installed by other parts of the app (or tests).
        self._hook_handles = [
            keyboard.on_press_key(self._trigger_key, self._on_press, suppress=False),
            keyboard.on_release_key(self._trigger_key, self._on_release, suppress=False),
        ]
        self.on_status(f"Ready — hold {self.hotkey.upper()} to speak")

    def stop(self, wait: bool = True):
        self._active = False
        for handle in self._hook_handles:
            try:
                keyboard.unhook(handle)
            except Exception:
                pass
        self._hook_handles = []
        # Tear down any in-flight streaming session — its worker thread
        # would otherwise outlive the app if no key-release has fired.
        self.recorder.on_chunk = None
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        # Signal worker to exit after it drains current job. Sentinel = None.
        self._worker_stop.set()
        # Drop stale queued recordings and guarantee the sentinel is delivered
        # even when the bounded queue is full. Close any streaming sessions
        # attached to dropped jobs so their workers exit too.
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if isinstance(job, tuple) and len(job) >= 5 and job[4] is not None:
                try:
                    job[4].close()
                except Exception:
                    pass
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            # Should not happen after draining, but do not block shutdown.
            log.debug("Kunde inte lägga stoppsentinel i transkriberingskö")
        worker = self._worker_thread
        if wait and worker and worker.is_alive():
            worker.join(timeout=5.0)
            self._worker_thread = None

    # ----------------------------------------------------------------- private

    def _modifier_held(self) -> bool:
        """All required modifiers must be physically held right now."""
        if not self._modifiers:
            return True
        try:
            return all(keyboard.is_pressed(m) for m in self._modifiers)
        except Exception:
            return False

    def _on_press(self, _):
        if self._active and not self._recording and self._modifier_held():
            try:
                self._recording = True
                # Wire the audio thread to push RMS levels directly to the
                # UI indicator — replaces the 50 ms polling timer with an
                # event-driven path. Cleared in _on_release.
                if self.indicator is not None:
                    self.recorder.on_level = self.indicator.push_level
                else:
                    self.recorder.on_level = None
                # Start a streaming session if enabled and supported by the
                # active backend. Wire raw audio chunks straight to it so
                # inference overlaps with the hold; UI partials are throttled
                # by the indicator.
                self._stream = None
                self.recorder.on_chunk = None
                if self.streaming:
                    try:
                        on_partial = (self.indicator.show_partial
                                      if self.indicator is not None
                                      else None)
                        self._stream = self.transcriber.start_stream(
                            on_partial=on_partial,
                        )
                        if self._stream is not None:
                            self.recorder.on_chunk = self._stream.push_audio
                    except Exception as e:
                        log.warning("Streaming-start misslyckades, faller "
                                    "tillbaka till batch: %s", e)
                        self._stream = None
                        self.recorder.on_chunk = None
                self.recorder.start_with_preroll()
                sounds.play_start()
                self.on_status("Lyssnar…")
                if self.indicator:
                    self.indicator.show("Lyssnar…", state="listen",
                                        level_source=lambda: self.recorder.level)
            except Exception as e:
                self._recording = False
                self._stream = None
                self.recorder.on_chunk = None
                log.error("Mic start error: %s", e, exc_info=True)
                sounds.play_error()
                if self.indicator:
                    self.indicator.show(f"Mikrofonfel: {e}", state="error")
                    self.indicator.hide(delay_ms=3000)

    def _on_release(self, _):
        if not (self._active and self._recording):
            return
        self._recording = False
        # Detach the UI push callback before stop_fast so a late audio
        # callback can't redraw bars after we've switched to transcribe.
        self.recorder.on_level = None
        # Stop pushing audio to the streaming session. The session keeps
        # running until finalize() is called from the worker.
        self.recorder.on_chunk = None
        stream = self._stream
        self._stream = None
        sounds.play_stop()
        # Stop the stream cheaply and hand back the captured audio. Downmix
        # and resample happen in the worker — keeping this hook callback
        # under ~10 ms so Windows doesn't disable the low-level hook.
        # stop_fast_async() defers PortAudio teardown to a daemon thread
        # so the key-release path returns immediately; the next start()
        # joins it before opening a fresh stream.
        try:
            audio, channels, rate = self.recorder.stop_fast_async()
        except Exception as e:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            log.error("Audio stop error: %s", e, exc_info=True)
            sounds.play_error()
            if self.indicator:
                self.indicator.show("Mikrofonfel", state="error")
                self.indicator.hide(delay_ms=2500)
            self.on_status(f"Klar — håll {self.hotkey.upper()}")
            return

        # Reuse the running RMS maintained by the recorder — O(1).
        rms = self.recorder.rms()

        # Enqueue for the worker. Bounded queue: if full (previous job(s)
        # still being transcribed/polished), drop and tell the user.
        try:
            self._jobs.put_nowait((audio, channels, rate, rms, stream))
        except queue.Full:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            log.warning("Transkriberingskö full — hoppar över denna")
            self.on_status("Upptagen — vänta…")
            if self.indicator:
                self.indicator.show("Upptagen", state="error")
                self.indicator.hide(delay_ms=1500)
            return

        self.on_status("Transkriberar…")
        if self.indicator:
            self.indicator.show("Transkriberar…", state="transcribe")

    def _worker_loop(self):
        """Drain the job queue: finalize audio → transcribe → paste."""
        while not self._worker_stop.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:  # sentinel
                break
            try:
                self._process_job(*job)
            except Exception as e:
                log.error("Worker exception: %s", e, exc_info=True)

    def _process_job(self, audio_raw: np.ndarray, channels: int,
                     rate: int, rms: float, stream=None):
        # Gate on the raw, native-rate buffer. We don't need to downmix or
        # resample just to count samples / check RMS — the running RMS from
        # the audio callback is already correct, and length scales linearly
        # with the sample rate, so the same threshold applied to native
        # samples just needs to be rescaled.
        n_raw = audio_raw.shape[0] if audio_raw is not None else 0
        min_raw = int(MIN_AUDIO_SAMPLES * rate / 16000) if rate else MIN_AUDIO_SAMPLES
        log.info("Audio: %d native samples @ %d Hz, RMS=%.5f", n_raw, rate, rms)

        if n_raw < min_raw:
            log.info("Inspelning för kort (%d < %d native samples), ignorerar",
                     n_raw, min_raw)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self.on_status(f"Klar — håll {self.hotkey.upper()}")
            if self.indicator:
                self.indicator.hide(delay_ms=0)
            return

        if rms < self.min_rms:
            log.info("Inspelning för tyst (RMS=%.5f < %.5f), ignorerar",
                     rms, self.min_rms)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self.on_status(f"Inget hördes — håll {self.hotkey.upper()}")
            if self.indicator:
                self.indicator.show("Inget hördes", state="error")
                self.indicator.hide(delay_ms=1500)
            return

        if stream is not None:
            self._finalize_streaming(stream)
        else:
            audio = finalize_audio(audio_raw, channels, rate)
            self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray):
        try:
            if self._worker_stop.is_set() or not self._active:
                log.info("Hoppar över stale transkribering efter stopp")
                return
            log.info("Transkriberar %d samples...", len(audio))
            # Use local-only transcription so the result is pasted immediately;
            # LLM polish runs asynchronously in the background if enabled.
            text = self.transcriber.transcribe_local(audio)
            self._deliver_text(text)
        except Exception as e:
            self._report_error(e)

    def _finalize_streaming(self, stream):
        """Pull the final hypothesis from a streaming session and deliver it.

        The streaming worker has already been transcribing in parallel with
        the hold, so this call usually returns within tens of ms (either
        the cached partial or one short tail pass).
        """
        try:
            if self._worker_stop.is_set() or not self._active:
                log.info("Hoppar över stale streaming-finalize efter stopp")
                try:
                    stream.close()
                except Exception:
                    pass
                return
            log.info("Finalize streaming-session…")
            text = stream.finalize()
            self._deliver_text(text)
        except Exception as e:
            try:
                stream.close()
            except Exception:
                pass
            self._report_error(e)

    def _deliver_text(self, text: str):
        """Paste post-processed text and trigger async LLM polish if on.

        Shared finishing path for the batch and streaming flows. Bails out
        when the worker has been signalled to stop between transcribe and
        paste (the user may have closed the app or restarted the dictation
        loop in that window).
        """
        text = snippet_module.expand(text)
        log.info("Resultat klart (%s)", _text_meta(text))
        if text.strip():
            if self._worker_stop.is_set() or not self._active:
                log.info("Hoppar över paste från stale transkribering")
                return
            inject_text(
                text,
                active_modifiers=self._modifier_keys,
                strategy=self.paste_strategy,
                paste_threshold=self.paste_threshold,
            )
            self.on_status(f"Klistrad — håll {self.hotkey.upper()} igen")
            if self.indicator:
                self.indicator.show("Klistrad", state="done")
                self.indicator.hide(delay_ms=1800)
            # Launch async LLM polish if enabled
            if (self.transcriber.llm_enabled
                    and self.transcriber.llm_api_key
                    and text.strip()):
                t = threading.Thread(
                    target=self._polish_async,
                    args=(text,),
                    daemon=True,
                    name="llm-polish",
                )
                t.start()
        else:
            self.on_status(f"Inget hördes — håll {self.hotkey.upper()}")
            if self.indicator:
                self.indicator.show("Inget hördes", state="error")
                self.indicator.hide(delay_ms=1500)

    def _report_error(self, e: Exception):
        log.error("Transkribering misslyckades: %s", e, exc_info=True)
        self.on_status(f"Fel — håll {self.hotkey.upper()}")
        if self.indicator:
            # Long stack-trace strings push the indicator off-screen and
            # leak internal paths to the user. Show only the first line
            # of the first message, with a sane upper bound.
            err_label = type(e).__name__
            err_msg = str(e).splitlines()[0] if str(e) else ""
            if len(err_msg) > 80:
                err_msg = err_msg[:77] + "..."
            pretty = f"{err_label}: {err_msg}" if err_msg else err_label
            self.indicator.show(f"Fel: {pretty}", state="error")
            self.indicator.hide(delay_ms=5000)

    def _polish_async(self, local_text: str):
        """Run LLM polish in the background after local text has been pasted.

        If the polished result differs from the local text, the polished version
        is copied to the clipboard and a toast indicator is shown so the user
        can paste it manually.  Any exception is logged but never propagated —
        LLM polish must never crash the dictation loop.
        """
        try:
            if self._worker_stop.is_set():
                log.info("LLM-polish avbrutet: app stängs ned")
                return

            import pyperclip

            from auto_learn import record_correction
            from llm_polish import polish

            tr = self.transcriber
            result = polish(
                local_text,
                tr.llm_api_key,
                tr.llm_model,
                style=tr.style,
                custom_prompt=tr.custom_style_prompt,
            )

            if self._worker_stop.is_set():
                log.info("LLM-polish avbrutet efter svar: app stängs ned")
                return

            if result.changed:
                record_correction(local_text, result.text)
                pyperclip.copy(result.text)
                log.info("Resultat (LLM) klart (%dms, %s)",
                         result.latency_ms, _text_meta(result.text))
                if self.indicator:
                    self.indicator.show("LLM klart — i urklipp", state="done")
                    self.indicator.hide(delay_ms=3000)
            else:
                log.info("LLM-polish: ingen ändring (%dms)", result.latency_ms)
        except Exception as e:
            log.error("LLM-polish misslyckades: %s", e, exc_info=True)
