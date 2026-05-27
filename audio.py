import logging
import math
import threading
import time as time_module

import numpy as np
import sounddevice as sd

try:
    import soxr as _soxr
except ImportError:
    _soxr = None

from scipy.signal import resample_poly

log = logging.getLogger("freewispr")

TARGET_RATE = 16000  # Whisper expects 16 kHz

# Hard cap on a single recording. At 48 kHz mono float32 this is ~23 MB,
# at 48 kHz stereo ~46 MB. Prevents a stuck hotkey from filling RAM.
MAX_RECORD_SECONDS = 120

# Host API preference order on Windows (best first)
_API_PREF = ["WASAPI", "DirectSound", "MME"]

# Cached snapshots of sounddevice's device/host-api tables. They are static
# until the user plugs/unplugs a device, so re-querying on every keypress
# is wasted IO (sounddevice rebuilds them from PortAudio each call).
_devices_cache: list | None = None
_hostapis_cache: list | None = None


def _select_level_channel(audio: np.ndarray) -> np.ndarray:
    """Return the channel with highest RMS from mono/stereo callback data."""
    if audio.ndim <= 1 or audio.shape[1] <= 1:
        return audio.ravel()
    # Some Windows/USB devices expose a silent left channel and put the mic on
    # the right channel. Pick the loudest channel instead of assuming channel 0.
    sums = np.einsum("ij,ij->j", audio, audio)
    return audio[:, int(np.argmax(sums))]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert captured audio to mono without assuming the mic is channel 0."""
    if audio.ndim <= 1 or audio.shape[1] <= 1:
        return audio.ravel()
    sums = np.einsum("ij,ij->j", audio, audio)
    loudest = int(np.argmax(sums))
    if sums[loudest] > 0 and float(np.min(sums)) <= float(sums[loudest]) * 0.01:
        return audio[:, loudest]
    return audio.mean(axis=1, dtype=np.float32)


def _devices() -> list:
    global _devices_cache
    if _devices_cache is None:
        _devices_cache = list(sd.query_devices())
    return _devices_cache


def _hostapis() -> list:
    global _hostapis_cache
    if _hostapis_cache is None:
        _hostapis_cache = list(sd.query_hostapis())
    return _hostapis_cache


def invalidate_device_cache() -> None:
    """Clear cached device/host-api tables. Call after a device change."""
    global _devices_cache, _hostapis_cache
    _devices_cache = None
    _hostapis_cache = None


def _api_priority() -> dict[int, int]:
    """Map host-api index -> priority (lower = better)."""
    prio = {}
    for rank, pref in enumerate(_API_PREF):
        for i, api in enumerate(_hostapis()):
            if pref in api["name"]:
                prio[i] = rank
    return prio


def list_input_devices() -> list[dict]:
    """Return deduplicated input devices sorted by API preference for the UI."""
    prio = _api_priority()
    apis = {i: api["name"] for i, api in enumerate(_hostapis())}

    devices = []
    for i, dev in enumerate(_devices()):
        if dev["max_input_channels"] < 1:
            continue
        devices.append({
            "index": i,
            "name": dev["name"],
            "api": apis.get(dev["hostapi"], "?"),
            "rate": int(dev["default_samplerate"]),
            "channels": dev["max_input_channels"],
            "_rank": prio.get(dev["hostapi"], 99),
        })

    devices.sort(key=lambda d: d["_rank"])
    seen = set()
    unique = []
    for d in devices:
        if d["name"] not in seen:
            seen.add(d["name"])
            unique.append(d)
    return unique


def _find_device_by_name(name: str) -> list[dict]:
    """Find all device entries matching a name, sorted by API preference."""
    prio = _api_priority()
    apis = {i: api["name"] for i, api in enumerate(_hostapis())}
    matches = []
    for i, dev in enumerate(_devices()):
        if dev["max_input_channels"] < 1:
            continue
        if name in dev["name"]:
            matches.append({
                "index": i,
                "rate": int(dev["default_samplerate"]),
                "channels": dev["max_input_channels"],
                "api": apis.get(dev["hostapi"], "?"),
                "_rank": prio.get(dev["hostapi"], 99),
            })
    matches.sort(key=lambda d: d["_rank"])
    return matches


def _resample(audio: np.ndarray, orig_rate: int) -> np.ndarray:
    """Resample audio from orig_rate to 16 kHz.

    Uses soxr when available (much faster: ~5 ms vs ~50 ms for 10s audio at
    44100 Hz). Falls back to scipy.signal.resample_poly which recomputes the
    FIR filter on every call but is always available.
    """
    if orig_rate == TARGET_RATE:
        return audio
    if _soxr is not None:
        return _soxr.resample(audio, orig_rate, TARGET_RATE, quality="HQ").astype(np.float32)
    g = math.gcd(TARGET_RATE, orig_rate)
    up = TARGET_RATE // g
    down = orig_rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def _try_start(device: int, rate: int, channels: int, callback) -> sd.InputStream:
    """Try to open and start a stream. Raises on failure."""
    s = sd.InputStream(samplerate=rate, channels=channels,
                       dtype="float32", device=device, callback=callback)
    s.start()
    return s


class MicRecorder:
    """Records from mic while a hotkey is held."""

    def __init__(self, device: str | dict | None = None):
        """``device`` accepts:
          * ``None`` — auto-select first available input.
          * ``str``  — legacy: a device name substring.
          * ``dict`` — structured id ``{"name": str, "api": str, "index": int}``;
            we try the exact (api, index) first, then fall back to name match
            so reordered USB devices still work.
        """
        # Pre-allocated ring buffer — written by the audio callback, read by
        # the worker on stop. Avoids per-callback np.copy + final concat pass.
        self._buffer: np.ndarray | None = None  # shape (capacity, channels) or (capacity,)
        self._buffer_capacity = 0
        self._buffer_offset = 0
        self._buffer_channels = 1
        self._buffer_overflow = False
        self.level = 0.0  # Current RMS level for UI visualization
        # Running sum of squares — lets stop() return RMS in O(1) without
        # the caller doing another full pass over the audio buffer.
        self._sumsq = 0.0
        self._sumsq_count = 0
        self.recording = False
        self._stream: sd.InputStream | None = None
        # Background thread that owns stream.stop()+close() after stop_fast_async.
        # Held only so start() can join it before opening a new stream — without
        # joining, PortAudio can race and the second start() either errors out
        # or grabs a stale device handle. The thread itself is daemon.
        self._pending_close: threading.Thread | None = None
        # ---- Pre-roll ring buffer (PR 2.9a) ------------------------------- #
        # When enabled, the audio callback also writes every chunk into a
        # short rolling buffer (~500 ms). At hotkey-press time, the most
        # recent N samples are copied into the head of self._buffer and
        # recording continues seamlessly — start-of-utterance consonants
        # that used to be clipped now make it through.
        #
        # The ring lives on the recorder so it survives across multiple
        # start()/stop_fast_async() cycles; only enable_preroll() /
        # disable_preroll() change its lifetime.
        self._preroll_enabled = False
        self._preroll_seconds = 0.5
        self._preroll_ring: np.ndarray | None = None
        self._preroll_capacity = 0  # samples-per-channel
        self._preroll_offset = 0    # next write index into ring
        self._preroll_full = False  # has the ring wrapped at least once
        self._preroll_channels = 1
        # Optional callback fired from the audio thread with the latest RMS
        # level (float). Used by the UI to drive the equalizer without a
        # polling timer. Must be cheap + thread-safe; the indicator throttles
        # and marshals to Tk.
        self.on_level = None  # type: ignore[assignment]
        # Normalise to (name, api, index) tuple regardless of input shape.
        if isinstance(device, dict):
            self._device_name = device.get("name") or None
            self._device_api = device.get("api") or None
            self._device_index = device.get("index")
            if not isinstance(self._device_index, int):
                self._device_index = None
        else:
            self._device_name = device  # str | None
            self._device_api = None
            self._device_index = None
        self._last_status_log = 0.0

    def start(self):
        """Start recording. Tries multiple device/channel combos until one works."""
        # If a previous stop_fast_async() handed cleanup to a background
        # thread, wait for it here — opening a new InputStream while the
        # old one is still being torn down races inside PortAudio and can
        # leak handles or hang the next start. The join is short (typically
        # 10-50 ms) and only happens on back-to-back dictations.
        pending = self._pending_close
        if pending is not None and pending.is_alive():
            pending.join(timeout=0.5)
        self._pending_close = None

        # Defensive: if a previous start() never reached stop() (e.g. exception
        # in the caller), close the leaked stream before opening a new one.
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self.level = 0.0
        self._sumsq = 0.0
        self._sumsq_count = 0
        self._buffer_offset = 0
        self._buffer_overflow = False
        self.recording = True

        candidates = self._build_candidates()
        last_err = None

        for dev_idx, rate, ch, label in candidates:
            try:
                # Pre-allocate buffer for this (rate, channels). Reuse across
                # subsequent starts when shape matches — saves an allocation.
                self._ensure_buffer(rate, ch)
                self._stream = _try_start(dev_idx, rate, ch, self._cb)
                self._rate = rate
                log.info("Inspelning startad: %s (dev=%d, %dHz, %dch)",
                         label, dev_idx, rate, ch)
                return
            except Exception as e:
                last_err = e

        raise last_err or RuntimeError("Ingen mikrofon kunde oppnas")

    def _ensure_buffer(self, rate: int, channels: int) -> None:
        """Allocate or re-allocate the ring buffer for (rate, channels)."""
        capacity = rate * MAX_RECORD_SECONDS
        if (self._buffer is None
                or self._buffer_capacity != capacity
                or self._buffer_channels != channels):
            if channels > 1:
                self._buffer = np.empty((capacity, channels), dtype=np.float32)
            else:
                self._buffer = np.empty(capacity, dtype=np.float32)
            self._buffer_capacity = capacity
            self._buffer_channels = channels

    # ---- Pre-roll (PR 2.9a) ---------------------------------------------- #

    def _ensure_preroll_ring(self, rate: int, channels: int, seconds: float) -> None:
        """(Re-)allocate the pre-roll ring buffer for the current shape.

        Sizing: seconds * rate samples per channel. Re-allocates and
        clears state when shape or duration changes.
        """
        capacity = max(1, int(seconds * rate))
        if (self._preroll_ring is None
                or self._preroll_capacity != capacity
                or self._preroll_channels != channels):
            if channels > 1:
                self._preroll_ring = np.zeros((capacity, channels), dtype=np.float32)
            else:
                self._preroll_ring = np.zeros(capacity, dtype=np.float32)
            self._preroll_capacity = capacity
            self._preroll_channels = channels
            self._preroll_offset = 0
            self._preroll_full = False

    def _write_preroll(self, indata: np.ndarray, n: int) -> None:
        """Wrap-around write of the latest callback chunk into the ring.

        Called from the audio thread. Must be cheap and allocation-free.
        """
        if self._preroll_ring is None or self._preroll_capacity <= 0:
            return
        # Match shape: ring is 1-D for mono, 2-D for multi-channel.
        if self._preroll_channels > 1:
            chunk = indata[:n]
        else:
            chunk = indata[:n].ravel() if indata.ndim > 1 else indata[:n]
        cap = self._preroll_capacity
        off = self._preroll_offset
        end = off + n
        if end <= cap:
            self._preroll_ring[off:end] = chunk
        else:
            # Wrap: write tail, then head.
            first = cap - off
            self._preroll_ring[off:cap] = chunk[:first]
            self._preroll_ring[0:n - first] = chunk[first:]
            self._preroll_full = True
        self._preroll_offset = (off + n) % cap
        if self._preroll_offset == 0 and n > 0:
            self._preroll_full = True

    def _drain_preroll(self) -> np.ndarray:
        """Return the ring contents in chronological order (oldest first).

        Returns an empty array if pre-roll has never received audio.
        Does NOT clear the ring; the caller (start_with_preroll) just
        copies into the main buffer.
        """
        if self._preroll_ring is None or self._preroll_capacity == 0:
            return np.empty(0, dtype=np.float32)
        off = self._preroll_offset
        if not self._preroll_full:
            # Only the [0:off] portion has been written so far.
            if off == 0:
                return np.empty(0, dtype=np.float32)
            return self._preroll_ring[:off].copy()
        # Ring has wrapped — chronological order is [off:] then [:off].
        return np.concatenate([self._preroll_ring[off:], self._preroll_ring[:off]])

    def enable_preroll(self, seconds: float = 0.5) -> None:
        """Arm the pre-roll ring buffer and start the underlying stream.

        Idempotent — calling while already enabled is a cheap no-op (only
        re-allocates the ring if seconds changed). The mic LED will stay
        lit for as long as pre-roll is on.
        """
        self._preroll_enabled = True
        self._preroll_seconds = max(0.05, float(seconds))
        # Determine the rate/channels we'll actually use. If a stream is
        # already running (recording), match its shape; otherwise pick the
        # first valid candidate and start a passive listening stream.
        if self._stream is not None:
            rate = getattr(self, "_rate", TARGET_RATE)
            channels = self._buffer_channels
            self._ensure_preroll_ring(rate, channels, self._preroll_seconds)
            return
        # Spin up a passive stream just for pre-roll.
        candidates = self._build_candidates()
        last_err: Exception | None = None
        for dev_idx, rate, ch, label in candidates:
            try:
                self._ensure_preroll_ring(rate, ch, self._preroll_seconds)
                self._preroll_channels = ch
                self._buffer_channels = ch  # so a later start() picks same shape
                self._stream = _try_start(dev_idx, rate, ch, self._cb)
                self._rate = rate
                log.info("Pre-roll aktiverad: %s (dev=%d, %dHz, %dch, %.0f ms)",
                         label, dev_idx, rate, ch, self._preroll_seconds * 1000)
                return
            except Exception as e:
                last_err = e
        self._preroll_enabled = False
        raise last_err or RuntimeError("Ingen mikrofon kunde oppnas for pre-roll")

    def disable_preroll(self) -> None:
        """Stop the passive listening stream and release the ring."""
        was_recording = self.recording
        self._preroll_enabled = False
        self._preroll_ring = None
        self._preroll_capacity = 0
        self._preroll_offset = 0
        self._preroll_full = False
        # If we're not actively recording, tear down the passive stream
        # so the mic LED turns off.
        if not was_recording and self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        log.info("Pre-roll inaktiverad")

    def start_with_preroll(self) -> None:
        """Start an active recording, seeded with the latest pre-roll audio.

        When pre-roll is not enabled or has no captured samples yet,
        behaves exactly like :py:meth:`start`.
        """
        if not self._preroll_enabled or self._stream is None:
            self.start()
            return
        # Pre-roll path: reuse the already-open stream, just flip state
        # and seed self._buffer with the ring contents.
        self.level = 0.0
        self._sumsq = 0.0
        self._sumsq_count = 0
        self._buffer_overflow = False
        rate = getattr(self, "_rate", TARGET_RATE)
        self._ensure_buffer(rate, self._preroll_channels)
        ring = self._drain_preroll()
        n = ring.shape[0]
        if n > 0 and self._buffer is not None:
            if self._buffer_channels > 1:
                # Ring already has the (n, ch) shape.
                self._buffer[:n] = ring
            else:
                self._buffer[:n] = ring
            self._buffer_offset = n
            # Seed the RMS accumulators so .rms() reflects pre-roll too.
            flat = ring.ravel() if ring.ndim > 1 else ring
            self._sumsq = float(np.dot(flat, flat))
            self._sumsq_count = flat.size
        else:
            self._buffer_offset = 0
        # Order matters: recording=True AFTER seeding so the callback
        # cannot append to _buffer mid-seed.
        self.recording = True

    def _build_candidates(self) -> list[tuple[int, int, int, str]]:
        """Build ordered list of (device_idx, rate, channels, label) to try."""
        candidates = []
        devs = _devices()
        apis = {i: api["name"] for i, api in enumerate(_hostapis())}

        # Highest priority: exact (api, index) match from a structured config.
        # This survives the device being renamed but not reordered.
        if self._device_index is not None and 0 <= self._device_index < len(devs):
            d = devs[self._device_index]
            if d["max_input_channels"] >= 1:
                api_name = apis.get(d["hostapi"], "?")
                if not self._device_api or api_name == self._device_api:
                    rate = int(d["default_samplerate"])
                    for ch in [1, d["max_input_channels"]]:
                        candidates.append((self._device_index, rate, ch,
                                           f"{d['name']} [{api_name}] (saved index)"))

        if self._device_name:
            # Fall back to name substring match across all APIs.
            for m in _find_device_by_name(self._device_name):
                name = devs[m["index"]]["name"]
                for ch in [1, m["channels"]]:
                    entry = (m["index"], m["rate"], ch,
                             f"{name} [{m['api']}]")
                    if entry not in candidates:
                        candidates.append(entry)

        # Then try all input devices sorted by API preference
        prio = _api_priority()
        all_devs = []
        for i, dev in enumerate(devs):
            if dev["max_input_channels"] < 1:
                continue
            rank = prio.get(dev["hostapi"], 99)
            all_devs.append((rank, i, dev))
        all_devs.sort(key=lambda x: x[0])

        for _, i, dev in all_devs:
            rate = int(dev["default_samplerate"])
            for ch in [1, dev["max_input_channels"]]:
                label = f"{dev['name']} (auto)"
                entry = (i, rate, ch, label)
                if entry not in candidates:
                    candidates.append(entry)

        return candidates

    def _cb(self, indata, frames, time, status):
        if status:
            now = time_module.monotonic()
            if now - self._last_status_log > 5.0:
                log.warning("Audio callback-status: %s", status)
                self._last_status_log = now
        n = indata.shape[0]
        if n <= 0:
            return

        # Always update the pre-roll ring if enabled, even when we are not
        # actively recording — that is the entire point.
        if self._preroll_enabled and self._preroll_ring is not None:
            self._write_preroll(indata, n)

        if not self.recording or self._buffer is None:
            return
        remaining = self._buffer_capacity - self._buffer_offset
        if remaining <= 0:
            if not self._buffer_overflow:
                log.warning("Inspelning naadde %d s max-cap, slutar buffra",
                            MAX_RECORD_SECONDS)
                self._buffer_overflow = True
            return
        n = min(n, remaining)
        # Copy directly into the pre-allocated arena — no per-callback
        # np.ndarray allocation, no final concat pass in stop_fast().
        if self._buffer_channels > 1:
            chunk = indata[:n]
            self._buffer[self._buffer_offset:self._buffer_offset + n] = chunk
            chunk_for_level = _select_level_channel(chunk)
        else:
            chunk_for_level = indata[:n].ravel() if indata.ndim > 1 else indata[:n]
            self._buffer[self._buffer_offset:self._buffer_offset + n] = chunk_for_level
        self._buffer_offset += n
        # Update level/RMS using the data we already have in cache.
        if chunk_for_level.size:
            ss = float(np.dot(chunk_for_level, chunk_for_level))
            self._sumsq += ss
            self._sumsq_count += chunk_for_level.size
            self.level = float(np.sqrt(ss / chunk_for_level.size))
            cb = self.on_level
            if cb is not None:
                try:
                    cb(self.level)
                except Exception:
                    # Never let a UI callback take down the audio thread.
                    pass

    def rms(self) -> float:
        """Return RMS of all captured audio so far.

        Computed from the running sum-of-squares maintained by the audio
        callback — O(1) regardless of recording length, so the caller
        avoids a second pass over the buffer.
        """
        if self._sumsq_count == 0:
            return 0.0
        return float(np.sqrt(self._sumsq / self._sumsq_count))

    def stop_fast(self) -> tuple[np.ndarray, int, int]:
        """Stop the stream cheaply and hand back the captured audio.

        Returns ``(audio_copy, channels, rate)``. ``audio_copy`` is a
        contiguous *copy* of the recorded samples (worst case ~23 MB at
        120 s @ 48 kHz mono) so the worker thread can safely process it
        while a new recording starts overwriting the ring buffer in place.

        Keeps the keyboard-hook thread responsive (Windows can disable a
        low-level hook that blocks > ~300 ms).
        """
        self.recording = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        rate = getattr(self, "_rate", TARGET_RATE)
        if self._buffer is None or self._buffer_offset == 0:
            return np.empty(0, dtype=np.float32), self._buffer_channels, rate
        view = self._buffer[:self._buffer_offset]
        # Copy so the worker can safely process while a new recording starts.
        # The copy is contiguous and ~23 MB worst case (120 s @ 48 kHz mono).
        captured = view.copy()
        return captured, self._buffer_channels, rate

    def stop_fast_async(self) -> tuple[np.ndarray, int, int]:
        """Same contract as :py:meth:`stop_fast` but defers PortAudio teardown.

        ``self._stream.stop()`` + ``close()`` together cost ~15-60 ms on
        Windows because they wait for the PortAudio callback thread to
        drain. That blocks the keyboard-hook thread on key-release, which
        the user perceives as paste latency.

        This variant grabs the audio buffer synchronously (cheap — just a
        np.copy), nulls out ``self._stream`` so subsequent code sees no
        active stream, and hands the actual stop()/close() pair to a
        daemon thread. The next :py:meth:`start` calls
        ``self._pending_close.join()`` so two streams never overlap.

        Falls back to a synchronous tear-down when no stream is active or
        if spawning the helper thread fails.

        When pre-roll is enabled, the stream is kept open (it has to keep
        feeding the ring), and only the ``recording`` flag is cleared —
        the audio is still snapshotted and returned synchronously.
        """
        self.recording = False
        rate = getattr(self, "_rate", TARGET_RATE)

        if self._buffer is None or self._buffer_offset == 0:
            captured: np.ndarray = np.empty(0, dtype=np.float32)
        else:
            captured = self._buffer[:self._buffer_offset].copy()

        if self._preroll_enabled and self._stream is not None:
            # Keep the stream alive — pre-roll needs it. Reset the buffer
            # offset so the next start_with_preroll() seeds cleanly.
            self._buffer_offset = 0
            return captured, self._buffer_channels, rate

        stream = self._stream
        self._stream = None

        if stream is None:
            return captured, self._buffer_channels, rate

        def _close():
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        try:
            t = threading.Thread(
                target=_close,
                name="audio-stream-close",
                daemon=True,
            )
            t.start()
            self._pending_close = t
        except Exception:
            # If we cannot spawn the helper for any reason, do it inline.
            _close()
            self._pending_close = None

        return captured, self._buffer_channels, rate

    def stop(self) -> np.ndarray:
        """Backward-compatible: stop + finalize in one call.

        Prefer :py:meth:`stop_fast` + :py:func:`finalize_audio` for the
        latency-sensitive dictation path.
        """
        audio, channels, rate = self.stop_fast()
        return finalize_audio(audio, channels, rate)


def finalize_audio(audio: np.ndarray, channels: int, orig_rate: int) -> np.ndarray:
    """Downmix to mono and resample to 16 kHz.

    Pulled out of ``MicRecorder.stop`` so the keyboard-hook thread can hand
    raw audio to a worker thread for processing. On a 30 s recording at
    48 kHz this work takes ~20-80 ms and must not block the hook callback.

    Accepts either a 1-D mono buffer or a 2-D ``(samples, channels)`` array;
    multi-channel input is converted to mono by selecting a lone active
    channel (common with USB mics) or averaging balanced stereo channels.
    """
    if audio is None or audio.size == 0:
        return np.array([], dtype=np.float32)

    if audio.ndim > 1 and audio.shape[1] > 1:
        mono = np.ascontiguousarray(_to_mono(audio))
    else:
        mono = audio.ravel()

    log.info("Rå audio: shape=%s, dtype=%s, rate=%d, peak=%.4f",
             mono.shape, mono.dtype, orig_rate,
             float(np.abs(mono).max()) if mono.size else 0.0)
    resampled = _resample(mono, orig_rate)
    log.info("Resamplerad: %d -> %d samples (%d->%dHz), peak=%.4f",
             len(mono), len(resampled), orig_rate, TARGET_RATE,
             float(np.abs(resampled).max()) if resampled.size else 0.0)
    return resampled
