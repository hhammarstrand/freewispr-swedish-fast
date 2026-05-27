"""
Audio feedback — synthesized pop sounds for recording start/stop.
Uses numpy + wave + winsound (all available: numpy from whisper, rest stdlib).
Sounds are generated once at import and cached as in-memory WAV bytes.
"""
import io
import logging
import wave

import numpy as np

log = logging.getLogger("freewispr")

SAMPLE_RATE = 22050


def _make_wav(samples: np.ndarray) -> bytes:
    """Convert float32 samples [-1, 1] to WAV bytes."""
    # Clip and convert to int16
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _generate_pop(freq_start: float, freq_end: float,
                  duration_ms: int = 60, volume: float = 0.25) -> bytes:
    """Generate a short pop/click sound with frequency sweep and smooth envelope.

    Args:
        freq_start: Starting frequency in Hz.
        freq_end: Ending frequency in Hz.
        duration_ms: Duration in milliseconds.
        volume: Peak amplitude (0.0 - 1.0).
    """
    n = int(SAMPLE_RATE * duration_ms / 1000)

    # Frequency sweep (linear)
    freq = np.linspace(freq_start, freq_end, n)
    # Phase integral for smooth sweep
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    tone = np.sin(phase)

    # Smooth envelope: quick attack, smooth decay (Hann-like)
    envelope = np.sin(np.linspace(0, np.pi, n)) ** 2

    samples = tone * envelope * volume
    return _make_wav(samples.astype(np.float32))


# Pre-generate sounds at module load (fast, ~1ms)
_SND_START = _generate_pop(freq_start=600, freq_end=1100, duration_ms=55, volume=0.20)
_SND_STOP = _generate_pop(freq_start=1100, freq_end=600, duration_ms=55, volume=0.20)
_SND_ERROR = _generate_pop(freq_start=300, freq_end=200, duration_ms=90, volume=0.25)


def play_start():
    """Play a soft rising pop (recording started)."""
    _play(_SND_START)


def play_stop():
    """Play a soft falling pop (recording stopped)."""
    _play(_SND_STOP)


def play_error():
    """Play a low thud (error occurred)."""
    _play(_SND_ERROR)


def _play(wav_bytes: bytes):
    """Play WAV bytes asynchronously (non-blocking)."""
    try:
        import winsound
        winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
    except Exception as e:
        # Non-critical — just skip sound on failure
        log.debug("Kunde inte spela ljud: %s", e)
