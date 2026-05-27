# Bench: Parakeet vs Whisper (Swedish)

Compares `nvidia/parakeet-tdt-0.6b-v3` against `faster-whisper-small` and
`faster-whisper-medium` on Mozilla Common Voice 17.0 Swedish clips.

## Methodology

- **Data:** First N validated clips from CV 17.0 `sv-SE` test split, resampled
  to 16k mono WAV.
- **Warm-up:** Each model transcribes clip 0 once before timed runs (excludes
  CUDA init, JIT compile, weight-load-to-GPU costs).
- **Latency:** Wall-clock `time.perf_counter()` around the inference call only
  (no audio I/O, no model loading).
- **WER:** Computed via `jiwer.wer(refs, hyps)` on raw text (no normalization
  applied — this is intentionally conservative; both backends will lose
  equally on casing and punctuation).
- **RTF:** Real-time factor = total inference time / total audio duration.
  Lower is better; <1.0 means faster-than-realtime.

## How to run

```pwsh
# One-time setup (~6GB download into .venv-parakeet, isolated from main app):
pwsh -File scripts/setup_parakeet_venv.ps1

# Activate the bench venv:
.\.venv-parakeet\Scripts\Activate.ps1

# CUDA run (RTX 4070):
python scripts/bench_parakeet.py --n 10 --device cuda

# CPU run for comparison:
python scripts/bench_parakeet.py --n 10 --device cpu
```

Results are appended to `bench/results.md`.

## Decision criteria

Parakeet wins if **either**:
- WER is materially lower (>2 pp absolute) at comparable latency, OR
- RTF is materially lower (<0.5x) at comparable WER

Otherwise we keep faster-whisper and pivot VÅG 3 toward UX polish.

## Results

See `bench/results.md` (gitignored; regenerated per run).
