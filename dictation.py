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
                 paste_threshold: int = 200):
        self.transcriber = transcriber
        self.hotkey = hotkey
        # MicRecorder accepts str (legacy), dict (structured), or None.
        self.recorder = MicRecorder(device=mic_device)
        self.on_status = on_status or (lambda msg: None)
        self.indicator = indicator
        self.min_rms = min_rms
        self.paste_strategy = paste_strategy
        self.paste_threshold = int(paste_threshold)
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
        # Signal worker to exit after it drains current job. Sentinel = None.
        self._worker_stop.set()
        # Drop stale queued recordings and guarantee the sentinel is delivered
        # even when the bounded queue is full.
        while True:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
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
                self.recorder.start_with_preroll()
                sounds.play_start()
                self.on_status("Lyssnar…")
                if self.indicator:
                    self.indicator.show("Lyssnar…", state="listen",
                                        level_source=lambda: self.recorder.level)
            except Exception as e:
                self._recording = False
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
            self._jobs.put_nowait((audio, channels, rate, rms))
        except queue.Full:
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
                     rate: int, rms: float):
        # Finalize off-hook: downmix + resample
        audio = finalize_audio(audio_raw, channels, rate)
        n = len(audio)
        log.info("Audio: %d samples, RMS=%.5f", n, rms)

        if n < MIN_AUDIO_SAMPLES:
            log.info("Inspelning för kort (%d samples), ignorerar", n)
            self.on_status(f"Klar — håll {self.hotkey.upper()}")
            if self.indicator:
                self.indicator.hide(delay_ms=0)
            return

        if rms < self.min_rms:
            log.info("Inspelning för tyst (RMS=%.5f < %.5f), ignorerar",
                     rms, self.min_rms)
            self.on_status(f"Inget hördes — håll {self.hotkey.upper()}")
            if self.indicator:
                self.indicator.show("Inget hördes", state="error")
                self.indicator.hide(delay_ms=1500)
            return

        self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray):
        try:
            if self._worker_stop.is_set() or not self._active:
                log.info("Hoppar över stale transkribering efter stopp")
                return
            log.info("Transkriberar %d samples...", len(audio))
            text = self.transcriber.transcribe(audio)
            # Apply snippet expansion — if full text is a trigger, replace it
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
            else:
                self.on_status(f"Inget hördes — håll {self.hotkey.upper()}")
                if self.indicator:
                    self.indicator.show("Inget hördes", state="error")
                    self.indicator.hide(delay_ms=1500)
        except Exception as e:
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
