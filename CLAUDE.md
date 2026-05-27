# CLAUDE.md

## Goal

Gör freewispr-fast till den snabbaste och mest pålitliga svenska speech-to-text-appen för Windows — med perceived latency under 200 ms (record→paste) på GPU, och en ren kodbas redo för publik release.

### Tre spår, i prioritetsordning:

**1. Parakeet-integration (backend-byte)**
Byt primär STT-backend från faster-whisper till NVIDIA Parakeet (parakeet-tdt-0.6b-v3). Benchmarks visar 4x snabbare latens med jämförbar WER. Behåll faster-whisper som fallback (CPU, egennamn). Abstrahera transcriber-lagret så backends kan bytas från tray-menyn.

**2. Latens och prestanda**
Driv ner end-to-end-latens: async LLM-polish efter paste, pre-allokerad audio-ringbuffer, soxr-resampling, kanaldetektering vid stream-open. Mät och rapportera perceived latency i benchmarks.

**3. Kodkvalitet och release-readiness**
Splitta ui.py i moduler, extrahera JsonCache-helper, uppdatera README och SPEC.md, pinna modellrevisioner med checksums, lägg till security scanning i CI, och dokumentera privacy/clipboard-beteende ordentligt.

## Arkitektur

- **main.py** — Entry point: systemfält, threading, app-livscykel
- **transcriber.py** — KB-Whisper + CUDA + decoder-optimeringar + hotwords
- **dictation.py** — Dikteringslogik: tangent → spela in → transkribera → klistra
- **audio.py** — Mikrofoninspelning (WASAPI-prio, resample, enhetsval)
- **paste.py** — Urklipp + keyboard.send (modifier pre-release)
- **ui.py** — Tkinter: flytande indikator, inställningar, snippets, ordlista
- **config.py** — JSON-konfiguration (`~/.freewispr-swedish-parakeet/config.json`)
- **corrections.py** — Personliga ordrättningar
- **snippets.py** — Textmallar/expansion
- **llm_polish.py** — Valfri LLM-granskning av transkriberad text
- **auto_learn.py** — Lär från LLM-diff till lokala korrektioner
- **json_store.py** — Atomisk JSON-lagring (tempfil + replace)

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
