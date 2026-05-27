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
        return wer(refs, self.hypotheses)

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
    """Load first N validated Swedish clips from Common Voice 17.0."""
    print(f"[data] Loading Common Voice 17.0 Swedish (n={n}) ...")
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "sv-SE",
        split="test",
        cache_dir=str(cache_dir),
        trust_remote_code=False,
    )
    samples: list[Sample] = []
    audio_dir = cache_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        if len(samples) >= n:
            break
        audio = row["audio"]  # {'array': np.ndarray, 'sampling_rate': int, 'path': str}
        ref = row["sentence"].strip()
        if not ref:
            continue
        # Resample to 16k mono and save as WAV (both models expect 16k).
        arr = audio["array"]
        sr = audio["sampling_rate"]
        if sr != 16000:
            import numpy as np
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
        "--skip",
        nargs="*",
        default=[],
        choices=["parakeet", "whisper-small", "whisper-medium"],
        help="skip specific backends",
    )
    args = ap.parse_args()

    samples = load_swedish_samples(args.n, args.cache)
    refs = [s.reference for s in samples]

    results: list[Result] = []
    if "parakeet" not in args.skip:
        results.append(bench_parakeet(samples, args.device))
    if "whisper-small" not in args.skip:
        results.append(bench_whisper(samples, "small", args.device))
    if "whisper-medium" not in args.skip:
        results.append(bench_whisper(samples, "medium", args.device))

    report = print_report(refs, results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(f"\n## Run device={args.device} n={args.n}\n")
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
