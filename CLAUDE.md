# CLAUDE.md

## Goal

Driv perceived latency (key-release → text in target app) under 100 ms på GPU genom att streama Parakeet-inferens under inspelning. Visa partiell hypotes live i flytande indikator. Behåll Whisper som batch-fallback. Bibehåll ren kodbas och publik release-readiness.

### Tre spår, i prioritetsordning:

**1. Streaming-inferens (Parakeet)**
Lägg till `start_stream()` på `ParakeetBackend` som tar emot 16 kHz mono-chunks under inspelning och returnerar växande partiella hypoteser. Implementeras först som chunked-batch (kör `model.transcribe()` på växande fönster var ~300 ms — RTF 0.013 ger gott om huvud) med möjlig uppgradering till NeMos cache-aware streaming om checkpointen stöder det. Vid key-release: `finalize()` returnerar slutlig hypotes, vanligen efter <50 ms eftersom merparten av audion redan transkriberats. Bakom config-flagga `streaming: bool` (default `false` tills bench validerar).

**2. UX för streaming**
`audio.MicRecorder` exponerar `on_chunk: Callable[[ndarray, int], None]` (50 ms granularitet, mono raw). `dictation.DictationMode._on_press` kopplar chunk-callbacken till en `StreamingSession` från transcriber-lagret. `ui.indicator` får `show_partial(text)` som ritar en sublinje under nivåstaplarna (throttle ~5 Hz). En paste vid release — ingen live-rewrite i target-app.

**3. Mätning och regression-guard**
Utöka `scripts/bench_parakeet.py` med `--mode streaming` som spelar upp FLEURS-klipp i wall-clock och mäter `release_t → finalize_return_t`, inte inferenstid. Lägg en "Streaming"-rad i `bench/BENCH.md`. Lägg `tests/test_streaming.py` med en `FakeStreamingBackend` som emitterar deterministiska partials och verifierar att (a) endast en `inject_text` anropas, (b) finaltexten matchar batch-resultatet, (c) Whisper-vägen är oförändrad.

### Avgränsningar

- Ingen live-paste/rewrite i target-app (Wisprflow-stil) — risken för janky UX i externa appar är för stor.
- Whisper får inte streaming. Asymmetrin är OK — Whisper är CPU/egennamn-fallback, inte huvudvägen.
- Ingen modell-switch under pågående inspelning.

## Arkitektur

- **main.py** — Entry point: systemfält, threading, app-livscykel
- **transcriber.py** — Backend-abstraherad STT (Parakeet / Whisper), corrections, postprocess
- **parakeet_backend.py** — NVIDIA Parakeet NeMo-wrapper (valfri, GPU-only)
- **dictation.py** — Dikteringslogik: tangent → spela in → transkribera → klistra (async LLM-polish)
- **audio.py** — Mikrofoninspelning (WASAPI-prio, soxr/scipy-resample, enhetsval)
- **paste.py** — Urklipp + keyboard.send (modifier pre-release)
- **ui.py** — Tkinter: flytande indikator, inställningar, snippets, ordlista
- **config.py** — JSON-konfiguration (`~/.freewispr-swedish-parakeet/config.json`)
- **corrections.py** — Personliga ordrättningar (via JsonCache)
- **snippets.py** — Textmallar/expansion (via JsonCache)
- **llm_polish.py** — Valfri LLM-granskning av transkriberad text
- **auto_learn.py** — Lär från LLM-diff till lokala korrektioner
- **json_store.py** — Atomisk JSON-lagring (tempfil + replace) + JsonCache-klass

## Kommandon

```bash
# Kör appen (dev)
python main.py

# Tester
pytest

# Lint
ruff check .

# Bygg exe
build.bat
```

## Konventioner

- Språk i kod: engelska (variabelnamn, kommentarer, commit messages)
- Språk i UI/docs: svenska
- Tester i `tests/`, med mocks för Whisper, keyboard, clipboard, ljud, nätverk
- JSON-filer skrivs atomiskt via `json_store.save_json_atomic`
- API-nycklar via `keyring`, aldrig i config.json
- Appen är Windows-only (PortAudio/WASAPI, pystray, keyboard-hook)
