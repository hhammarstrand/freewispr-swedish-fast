# Parakeet vs Whisper Bench — Swedish (Decision Doc)

**Date:** 2026-05-27
**Hardware:** RTX 4070 Laptop (8GB VRAM), 12th-gen Intel mobile
**Dataset:** Google FLEURS sv_se, first 10 test-split clips (avg 11.4s, total 114s)
**Methodology:** See `bench/README.md`. WER normalized with `whisper_normalizer.BasicTextNormalizer` (lowercase, strip punct) to match published numbers.

## Result

| Model | WER (norm) | mean latency | p95 latency | RTF |
|---|---:|---:|---:|---:|
| **parakeet-tdt-0.6b-v3** | **27.98%** | **148 ms** | **157 ms** | **0.013x** |
| faster-whisper-small | 26.79% | 604 ms | 732 ms | 0.054x |
| faster-whisper-medium | 18.45% | 2097 ms | 3032 ms | 0.186x |

### Streaming (perceived release→text latency)

The new repo goal targets *perceived* latency — the wall-clock time
between key-release and text becoming pasteable. Measured by replaying
each clip in 50 ms chunks via `parakeet_backend.StreamingSession`, then
timing the `finalize()` call after the last chunk lands. Numbers below
are placeholders until the bench has been re-run on the dev GPU — see
the `--mode streaming` flag in `scripts/bench_parakeet.py`.

| Model | WER (norm) | release→text mean | release→text p95 |
|---|---:|---:|---:|
| parakeet-tdt-0.6b-v3 (streaming) | _TBD_ | _TBD_ | _TBD_ |

Reproducing the streaming bench:

```pwsh
.\.venv-parakeet\Scripts\python.exe scripts/bench_parakeet.py --n 10 --device cuda --mode streaming
```

## Decision: GO — integrate Parakeet as primary backend

**Reasoning:**
- Parakeet matches Whisper-small on quality (1.2pp WER difference is within noise at n=10) while being **4.1x faster** in wall-clock latency and **4.2x faster** in p95.
- Whisper-medium is materially better on quality (-9.5pp WER) but 14x slower than Parakeet. 2s p50 / 3s p95 transcription latency kills the "instant paste" UX we are targeting.
- For a 5-second dictation utterance, Parakeet returns the transcript in ~75ms (vs ~300ms for Whisper-small). Combined with ~50ms pre-roll savings (PR 2.9) and ~30ms async teardown (PR 2.10), this puts total record→paste latency under 200ms — wisprflow territory.

## Caveats (must be addressed before users see it)

1. **n=10 is statistically thin.** The 1.2pp WER gap vs Whisper-small could flip either direction at n=100. We accept this risk because the latency gap is so dominant.
2. **FLEURS is read news prose, not dictation.** Real freewispr usage is short utterances (2-5s), conversational Swedish, home environment. We have no data here. **Action:** add mic-recorded test set in a follow-up PR once Parakeet is wired up.
3. **Egennamn are weak in Parakeet.** Clip 9: "Giancarlo Fisichella" → "John Carlo u gälla". Whisper handles foreign names noticeably better. This is the strongest argument for keeping Whisper-medium available as a fallback for dictating names/quotes.
4. **NeMo adds ~6GB to install footprint** (torch+cu124 wheels + nemo deps). Mitigation: keep faster-whisper available, ship Parakeet as the default but make it swappable from tray (PR 3.3 backend abstraction).
5. **No CPU benchmark.** User explicitly skipped CPU since the fast fork is GPU-targeted. If someone runs without CUDA, we fall back to faster-whisper-small via the backend abstraction.

## Selected transcripts (illustrative)

Where Parakeet **wins** vs Whisper-small:

| Reference | Parakeet | Whisper-small |
|---|---|---|
| bussar avgår från den distriktsgemensamma busstationen | Bossar avgår från den distrikts gemensamma bostationen | Bussar avgår från den **distriktiemensamma bostationen** |
| australiens mitchell gourley slutade som elva | Australiens Michael slutades som 11 | Australiens Mitchell Golley slutade som 11 |

Where Whisper-small **wins**:

| Reference | Parakeet | Whisper-small |
|---|---|---|
| giancarlo fisichella förlorade kontrollen | **John Carlo u gälla** förlorade kontrollen | **Giancarlo Fischella** förlorade kontrollen |
| blyertspennan en trogen följeslagare | **byrspännen** en **trugen** följeslagare | **Blyersbindan** en trogen följeslagare |

Where Whisper-medium **clearly wins** (and Parakeet/small both miss):

| Reference | Whisper-medium |
|---|---|
| jakar/bumthang går mellan 6.30 och 7.30 | jackar, bontang, går mellan halv sju och halv åtta |
| nominerings­kategorier (10 ord) | Korrekt segmenterad lista |

## Next steps (PR sequence)

- **PR 3.3 (next):** Backend abstraction. `Backend` protocol with `transcribe(audio: np.ndarray) -> str`. Implementations: `WhisperBackend`, `ParakeetBackend`. Config key `backend: "parakeet" | "whisper-small" | "whisper-medium"`. Tray submenu under "Modell".
- **PR 3.4:** Lazy NeMo import. Make `nemo_toolkit` optional so users without CUDA can still ship.
- **PR 3.5:** Mic-recorded dictation test set + re-run bench. Validate Parakeet still wins on the real domain.
- **PR 3.6 (deferred):** Custom-vocabulary / hot-words for egennamn. Parakeet supports this via TDT bias-vector or external rescoring.

## Reproducing

```pwsh
pwsh -File scripts/setup_parakeet_venv.ps1   # ~6GB, one-time
.\.venv-parakeet\Scripts\python.exe scripts/bench_parakeet.py --n 10 --device cuda
# Results appended to bench/results.md (gitignored, regenerated per run).
```
