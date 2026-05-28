"""
Bench: Parakeet-TDT-0.6b-v3 vs faster-whisper (small + medium) on Swedish.

Goal: decide whether to swap the Whisper backend for Parakeet in the fast fork.
Metrics:
  - WER (word error rate) vs reference transcript via jiwer
  - Latency: mean + p95 of pure model inference time per clip
  - Real-time factor (RTF) = inference_time / audio_duration

Test data: Mozilla Common Voice 17.0 Swedish, first N validated clips.
Each model warmed up once before timed runs to exclude JIT/CUDA-init cost.

Run from .venv-parakeet:
    python scripts/bench_parakeet.py --n 10 --device cuda
    python scripts/bench_parakeet.py --n 10 --device cpu

Streaming mode:
    python scripts/bench_parakeet.py --n 10 --device cuda --mode streaming

In streaming mode we measure *perceived* latency: how long from "user
released the hotkey" (i.e. the last audio chunk was pushed) to "final text
is available to paste". The clip is replayed in 50 ms chunks at wall-clock
pace so the Parakeet StreamingSession runs concurrently with the playback,
mimicking real dictation. This is the metric the new repo goal targets,
not the batch-inference time reported by the default mode.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from jiwer import wer
from whisper_normalizer.basic import BasicTextNormalizer

_NORMALIZER = BasicTextNormalizer()


def _normalize(text: str) -> str:
    """Strip case, punctuation, extra whitespace before WER.

    Mirrors what OpenAI / NeMo report in published WER numbers. Without this,
    'Hello, world.' vs 'hello world' counts as 100% WER which is misleading
    for ASR comparisons.
    """
    return _NORMALIZER(text).strip()


@dataclass
class Sample:
    audio_path: Path
    reference: str
    duration_s: float


@dataclass
class Result:
    model: str
    device: str
    hypotheses: list[str] = field(default_factory=list)
    latencies_s: list[float] = field(default_factory=list)
    audio_durations_s: list[float] = field(default_factory=list)

    def wer(self, refs: list[str]) -> float:
        return wer([_normalize(r) for r in refs], [_normalize(h) for h in self.hypotheses])

    def latency_mean(self) -> float:
        return statistics.mean(self.latencies_s)

    def latency_p95(self) -> float:
        s = sorted(self.latencies_s)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    def rtf(self) -> float:
        total_inf = sum(self.latencies_s)
        total_audio = sum(self.audio_durations_s)
        return total_inf / total_audio if total_audio > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def load_swedish_samples(n: int, cache_dir: Path) -> list[Sample]:
    """Load first N Swedish clips from Google FLEURS (sv_se).

    FLEURS is open (no HF auth required), CC-BY-4.0, ~3h per language with
    professional readers. Test split has ~700 clips averaging ~10s.

    We avoid HF's audio-decoder entirely (it requires torchcodec + FFmpeg DLLs
    on Windows) by loading the dataset without decoding and reading raw WAVs
    from the local cache with soundfile.
    """
    import numpy as np
    from datasets import Audio

    print(f"[data] Loading google/fleurs sv_se (n={n}) ...")
    ds = load_dataset(
        "google/fleurs",
        "sv_se",
        split="test",
        cache_dir=str(cache_dir),
    )
    # Disable HF's audio decoder — we'll read paths manually.
    ds = ds.cast_column("audio", Audio(decode=False))

    samples: list[Sample] = []
    audio_dir = cache_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        if len(samples) >= n:
            break
        ref = (row.get("transcription") or row.get("raw_transcription") or "").strip()
        if not ref:
            continue
        src_bytes = row["audio"]["bytes"]
        if not src_bytes:
            print(f"  [skip] missing audio bytes for row {i}")
            continue
        import io
        arr, sr = sf.read(io.BytesIO(src_bytes), dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            arr = resample_poly(arr, 16000, sr).astype(np.float32)
            sr = 16000
        path = audio_dir / f"sv_{i:03d}.wav"
        sf.write(path, arr, sr, subtype="PCM_16")
        samples.append(Sample(audio_path=path, reference=ref, duration_s=len(arr) / sr))
    print(f"[data] Prepared {len(samples)} clips, total {sum(s.duration_s for s in samples):.1f}s")
    return samples


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

def bench_parakeet(samples: list[Sample], device: str) -> Result:
    import nemo.collections.asr as nemo_asr

    print(f"[parakeet] Loading parakeet-tdt-0.6b-v3 on {device} ...")
    t0 = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model = model.to(device)
    model.eval()
    print(f"[parakeet] Loaded in {time.perf_counter() - t0:.1f}s")

    # Warm-up (compile + CUDA init).
    print("[parakeet] Warming up ...")
    _ = model.transcribe([str(samples[0].audio_path)])

    result = Result(model="parakeet-tdt-0.6b-v3", device=device)
    for s in samples:
        t = time.perf_counter()
        out = model.transcribe([str(s.audio_path)])
        dt = time.perf_counter() - t
        # NeMo 2.x returns list[Hypothesis] with .text; older returns list[str].
        hyp = out[0].text if hasattr(out[0], "text") else out[0]
        result.hypotheses.append(hyp.strip())
        result.latencies_s.append(dt)
        result.audio_durations_s.append(s.duration_s)
        print(f"  {s.audio_path.name}: {dt*1000:.0f}ms -> {hyp[:60]}")
    return result


def bench_parakeet_streaming(samples: list[Sample], device: str,
                             chunk_ms: int = 50) -> Result:
    """Wall-clock streaming bench.

    Replays each clip in real time, feeding ``chunk_ms`` chunks to a
    :class:`parakeet_backend.StreamingSession`. The recorded latency is the
    delta between the *last* ``push_audio`` call and ``finalize()`` returning
    — i.e. the latency the user perceives at key-release. Inference that
    happened during the hold is not counted (that's the whole point).

    We deliberately import the production ParakeetBackend here rather than
    talk to NeMo directly: the bench then exercises the same code path the
    app uses, including the chunked-batch worker and the tempfile-write
    that ``transcribe_raw`` currently relies on.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from parakeet_backend import ParakeetBackend  # type: ignore[import-not-found]

    print(f"[parakeet-stream] Loading backend on {device} (chunk={chunk_ms}ms) ...")
    t0 = time.perf_counter()
    backend = ParakeetBackend(device=device)
    print(f"[parakeet-stream] Loaded in {time.perf_counter() - t0:.1f}s")

    # Warm-up the streaming path so first-clip latency doesn't include
    # CUDA-init/JIT cost — same policy as the batch bench above.
    print("[parakeet-stream] Warming up ...")
    warm = backend.start_stream()
    arr0, _sr0 = sf.read(str(samples[0].audio_path), dtype="float32", always_2d=False)
    warm.push_audio(arr0[:1600], 16000)
    warm.finalize(timeout=10.0)

    result = Result(model="parakeet-tdt-0.6b-v3-streaming", device=device)
    for s in samples:
        arr, sr = sf.read(str(s.audio_path), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        chunk_n = max(1, int(sr * chunk_ms / 1000))

        sess = backend.start_stream()
        wall_t0 = time.perf_counter()
        # Replay at wall-clock pace — sleep between chunks so the streaming
        # worker has the same window to do its job as it would in production.
        for off in range(0, len(arr), chunk_n):
            sess.push_audio(arr[off:off + chunk_n], int(sr))
            target = wall_t0 + (off + chunk_n) / sr
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
        # "User releases hotkey" — start the latency clock NOW.
        release_t = time.perf_counter()
        hyp = sess.finalize(timeout=10.0)
        dt = time.perf_counter() - release_t

        result.hypotheses.append(hyp.strip())
        result.latencies_s.append(dt)
        result.audio_durations_s.append(s.duration_s)
        print(f"  {s.audio_path.name}: release→text {dt*1000:.0f}ms -> {hyp[:60]}")
    return result


def bench_whisper(samples: list[Sample], model_size: str, device: str) -> Result:
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[whisper] Loading faster-whisper {model_size} on {device} ({compute_type}) ...")
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"[whisper] Loaded in {time.perf_counter() - t0:.1f}s")

    # Warm-up.
    print("[whisper] Warming up ...")
    list(model.transcribe(str(samples[0].audio_path), language="sv")[0])

    result = Result(model=f"faster-whisper-{model_size}", device=device)
    for s in samples:
        t = time.perf_counter()
        segs, _info = model.transcribe(str(s.audio_path), language="sv", beam_size=5)
        text = " ".join(seg.text for seg in segs).strip()
        dt = time.perf_counter() - t
        result.hypotheses.append(text)
        result.latencies_s.append(dt)
        result.audio_durations_s.append(s.duration_s)
        print(f"  {s.audio_path.name}: {dt*1000:.0f}ms -> {text[:60]}")
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def print_report(refs: list[str], results: list[Result]) -> str:
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("WER computed on normalized text (BasicTextNormalizer: lowercase, no punct)")
    lines.append(f"{'model':<32}{'device':<8}{'WER':>8}{'mean ms':>10}{'p95 ms':>10}{'RTF':>8}")
    lines.append("-" * 78)
    for r in results:
        lines.append(
            f"{r.model:<32}{r.device:<8}"
            f"{r.wer(refs)*100:>7.2f}%"
            f"{r.latency_mean()*1000:>10.0f}"
            f"{r.latency_p95()*1000:>10.0f}"
            f"{r.rtf():>8.3f}"
        )
    lines.append("=" * 78)
    report = "\n".join(lines)
    print(report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of CV clips")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--cache", type=Path, default=Path("bench/cv_cache"))
    ap.add_argument("--out", type=Path, default=Path("bench/results.md"))
    ap.add_argument(
        "--mode",
        choices=["batch", "streaming"],
        default="batch",
        help="batch = pure inference latency; streaming = release→text latency "
             "with wall-clock replay (matches the production dictation flow)",
    )
    ap.add_argument(
        "--chunk-ms",
        type=int,
        default=50,
        help="streaming chunk size in milliseconds (matches audio.MicRecorder)",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["parakeet", "whisper-small", "whisper-medium"],
        help="skip specific backends (batch mode only)",
    )
    args = ap.parse_args()

    samples = load_swedish_samples(args.n, args.cache)
    refs = [s.reference for s in samples]

    results: list[Result] = []
    if args.mode == "streaming":
        # Streaming is Parakeet-only by goal — Whisper has no streaming path.
        results.append(bench_parakeet_streaming(samples, args.device,
                                                chunk_ms=args.chunk_ms))
    else:
        if "parakeet" not in args.skip:
            results.append(bench_parakeet(samples, args.device))
        if "whisper-small" not in args.skip:
            results.append(bench_whisper(samples, "small", args.device))
        if "whisper-medium" not in args.skip:
            results.append(bench_whisper(samples, "medium", args.device))

    report = print_report(refs, results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(f"\n## Run mode={args.mode} device={args.device} n={args.n}\n")
        f.write("```\n" + report + "\n```\n")
        f.write("\n### Reference vs hypotheses\n")
        for i, s in enumerate(samples):
            f.write(f"\n**Clip {i}** ({s.duration_s:.1f}s)\n")
            f.write(f"- REF: {s.reference}\n")
            for r in results:
                f.write(f"- {r.model}: {r.hypotheses[i]}\n")
    print(f"\n[bench] Appended results to {args.out}")


if __name__ == "__main__":
    main()
