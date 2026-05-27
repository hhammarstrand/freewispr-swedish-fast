# freewispr-fast

**Svensk speech-to-text diktering for Windows — optimerad for latens.**
Diktera var som helst. Lokal transkribering med Parakeet eller Whisper, med valfri asynkron LLM-granskning.

> **Fork av:** [x26prakhar/freewispr](https://github.com/x26prakhar/freewispr)
> **Licens:** MIT

---

## Vad ar freewispr-fast?

freewispr-fast ar en prestandafork av freewispr-swedish med fokus pa **lagsta mojliga latens** for svensk diktering. Appen stodjer tva STT-backends:

- **Parakeet** (nvidia/parakeet-tdt-0.6b-v3) — standard pa GPU, ~148 ms medellatens (4x snabbare an Whisper)
- **Whisper** (KBLab/kb-whisper-*) — fallback for CPU eller nar NeMo inte ar installerat

Backend kan bytas fran systemfaltets meny (Backend → Auto/Parakeet/Whisper).

---

## Funktionalitet

- **Diktering** — hall tangent nedtryckt, prata, slapp. Texten klistras in direkt vid markoren
- **Dubbel backend** — Parakeet (GPU, snabbast) eller KBLab Whisper (CPU/GPU, bast egennamn)
- **Asynkron LLM-granskning** — lokal text klistras in omedelbart; LLM-polish kor i bakgrunden och kopierar till urklipp om texten andras
- **Flytande indikator** — visar Lyssnar / Transkriberar / Klar / Fel med animerade equalizer-staplar
- **Ljudatergivning** — mjuka pop-ljud vid inspelningsstart och -stopp
- **Pre-roll** — valfri ~500 ms ringbuffer sa inledande konsonanter inte klipps
- **Hybrid textinmatning** — kort text via keyboard.write (~0 ms), langre via urklipp+Ctrl+V
- **Mikrofonval** — valj mikrofon i installningar (WASAPI, DirectSound, MME)
- **Tystnadsdetektion** — avvisar for tysta inspelningar automatiskt (RMS-baserad)
- **Personlig ordlista** — lagg till rattningar for ord som transkriberas fel
- **Hotwords** — mata in egna termer/namn som modellen ska prioritera
- **Snippets** — textmallar som expanderas automatiskt
- **Auto-larning** — LLM-korrigeringar laggs automatiskt i ordlistan efter upprepade forekomster
- **Flermodell-stod** — tiny, base, small, medium, large (alla KBLab Whisper)
- **Stilpresets** — vardaglig/formell/kod/e-post/egen prompt for LLM-granskning
- **Starta med Windows** — enkel toggle i menyn
- **Systemfack** — lever diskret i bakgrunden
- **Offline-lage** — Parakeet/Whisper-diktering sker helt lokalt
- **GPU-stod** — automatisk CUDA-detektion for NVIDIA-grafikkort

---

## Snabbstart

### Ladda ner (rekommenderas)

Ladda ner senaste releasen fran [**Releases**](https://github.com/hhammarstrand/freewispr-swedish/releases).

**Krav:** Windows 10 eller Windows 11

```
1. Ladda ner och packa upp freewispr-swedish-mappen
2. Kor freewispr-swedish.exe
3. En lila mikrofon-ikon visas i systemfacket
4. Hall Ctrl+Space och prata. Slapp for att klistra in.
```

> Vid forsta starten laddas KB-Whisper-small (~500 MB) ner automatiskt.
> Darefter fungerar Whisper-diktering offline sa lange LLM-granskning ar avstangd.

---

## Integritet och natverk

Som standard sker **all** transkribering lokalt — inget ljud eller text lamnar datorn.

- **Ljud stannar lokalt.** Parakeet och Whisper kor helt pa din maskin. Inget ljud skickas over natverk.
- **Modellnedladdning.** Forsta starten laddar ner modellen fran Hugging Face / NVIDIA (en gang). Darefter ar appen offline.
- **LLM-granskning ar AVSTANGD som standard.** Nar du aktiverar den skickas *transkriberad text* (aldrig ljud) till GitHub Models/Azure for korrigering.
- **Asynkron LLM-polish.** Nar LLM ar pa klistras lokal text in forst. LLM-resultatet kopieras till urklipp i bakgrunden — synligt via en toast-notis.
- **Urklipp.** Text klistras in via systemets urklipp. Appen forsoker aterstalla tidigare urklippsinnehall efter paste.
- **Loggning.** Dikterad text loggas inte som standard; loggen innehaller metadata (langd, modell, latency, status).
- **Lokal data.** Ordkorrektioner, snippets och inlarningsdata sparas i `~/.freewispr-swedish-parakeet/`. Rensa via menyval "Sekretess och data".
- **API-nycklar** sparas i Windows Credential Manager via `keyring`, aldrig i config-filen.

---

## Bygga fran kallkod

**Krav:** Python 3.10+, Windows 10/11

```bash
# Klona repot
git clone https://github.com/hhammarstrand/freewispr-swedish-fast.git
cd freewispr-swedish-fast

# Installera dependencies
pip install -r requirements.txt

# (Valfritt) Installera GPU-stod (NVIDIA, ~2.5 GB nedladdning)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# (Valfritt) Installera Parakeet-backend (kraver GPU, ~6 GB nedladdning)
pip install nemo_toolkit[asr]

# Kor direkt fran kallkod
python main.py

# Eller anvand run.bat (installerar beroenden automatiskt)
run.bat
```

### Bygga exe

```bash
# build.bat installerar beroenden och bygger med PyInstaller (--onedir)
build.bat
```

Bygget skapar en `dist/freewispr-swedish/`-mapp med `freewispr-swedish.exe` och alla beroenden.

---

## Backends

### Parakeet (standard med GPU)

Nar CUDA ar tillgangligt och `nemo_toolkit[asr]` ar installerat anvands NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v3`) som primar backend. Parakeet ger ~4x lagre latens jamfort med Whisper vid jamforbar WER.

Backend valjs fran systemfacksikonen -> **Backend** -> Auto/Parakeet/Whisper.

- **Auto** (standard) -- Parakeet om GPU + NeMo finns, annars Whisper
- **Parakeet** -- tvinga Parakeet (kraver NeMo + CUDA)
- **Whisper** -- tvinga Whisper oavsett GPU

### Whisper-modeller (KBLab)

[KBLab:s Whisper-modeller](https://huggingface.co/KBLab) anvands som fallback, tranade pa over 50 000 timmar svenskt tal.

| Modell | WER (svenska) | Storlek | Jamforelse OpenAI |
|--------|---------------|---------|-------------------|
| `tiny` | 13.2 | ~40 MB | 59.2 |
| `base` | 9.1 | ~150 MB | 39.6 |
| **`small`** | **7.3** | **~500 MB** | **20.6** |
| `medium` | 6.6 | ~1.5 GB | 15.8 |
| `large` | -- | ~3 GB | -- |

**Standard:** `small` -- basta balansen mellan hastighet och precision for svenska.

Modeller sparas i `~/.freewispr-swedish-parakeet/models/` och laddas ner vid forsta anvandning.

### Konvertera modeller (medium/large)

Medium- och large-modellerna kan behova konverteras till CTranslate2-format for att undvika vocabulary-krascher:

```bash
pip install ctranslate2 transformers
python convert_model.py medium
python convert_model.py large
```

Konverterade modeller sparas i `~/.freewispr-swedish-parakeet/models/kb-whisper-{size}-ct2/`.

---

## Installningar

Hogerklicka pa systemfacksikonen och valj **Installningar**.

- **Snabbtangent** — klicka och tryck valfri tangentkombination
- **Mikrofon** — valj specifik mikrofon eller "Auto"
- **Modell** — valj storlek (tiny/base/small/medium/large) for Whisper-fallback
- **GPU (CUDA)** — sla pa/av GPU-acceleration
- **Backend** — valj Auto/Parakeet/Whisper fran tray-menyn
- **Stil** — vardaglig/formell/kod/e-post/egen prompt for LLM-granskning
- **Pre-roll** — hall mikrofonen aktiv for snabbare start (mic-LED lyser)

---

## Hotwords (personlig ordlista for Whisper)

Hotwords ar termer, namn och fraser som Whisper ska prioritera vid transkribering. De forbattrar precision for ovanliga ord, egennamn och facktermer.

Hotwords hamtas fran tva kallor:

1. **Personlig ordlista** -- de korrekta varden du lagt in via systemfacket ("Personlig ordlista")
2. **hotwords.txt** -- valfri fil pa `~/.freewispr-swedish-parakeet/hotwords.txt`, ett ord/fras per rad

Exempel pa `hotwords.txt`:
```
# Egennamn
Prakhar
Hammarstrand

# Facktermer
CTranslate2
PyInstaller
```

---

## Mikrofoninspelning

Appen stodjer WASAPI, DirectSound och MME som audio-backends, med automatisk prioritering:

1. **WASAPI** (bast kvalitet, lagst latens)
2. **DirectSound** (bra kompatibilitet)
3. **MME** (bredast stod)

Inspelning sker i mikrofonens nativa samplerate (t.ex. 48 kHz) och resamplas till 16 kHz. Nar `soxr` ar installerat anvands det for resampling (~5 ms for 10 s ljud), annars faller appen tillbaka till `scipy.signal.resample_poly` (~50 ms). Flerkanaliga mikrofoner mixas till mono automatiskt (loudest-channel-val for USB-headsets med tyst kanal).

---

## Teknisk arkitektur

```
freewispr-fast/
+-- main.py              # Entry point: systemfack, threading, applifecycle
+-- transcriber.py       # Backend-abstraktion: Parakeet/Whisper + hotwords + postprocessing
+-- parakeet_backend.py  # NVIDIA Parakeet NeMo-wrapper (nvidia/parakeet-tdt-0.6b-v3)
+-- dictation.py         # Dikteringslogik: tangent -> spela in -> transkribera -> klistra
+-- audio.py             # Mikrofoninspelning (WASAPI prio, soxr/scipy resample, pre-roll)
+-- text_inject.py       # Hybrid textinjektion (keyboard.write for kort, clipboard for lang)
+-- paste.py             # Aldre clipboard-paste (bakatkompatiblitet)
+-- sounds.py            # Syntetiserade pop-ljud for inspelningsatergivning
+-- ui.py                # Tkinter: flytande indikator, installningar, snippets, ordlista
+-- model_ui.py          # Modellnedladdning och -hantering (forstagangs-dialog, modellhanterare)
+-- privacy_ui.py        # Integritets-/datarensnings-UI
+-- config.py            # JSON konfiguration (~/.freewispr-swedish-parakeet/config.json)
+-- json_store.py        # Atomisk JSON-lagring (tempfil + replace) + JsonCache-klass
+-- corrections.py       # Personliga ordrattningar (via JsonCache)
+-- snippets.py          # Textmallar/expansion (via JsonCache)
+-- llm_polish.py        # Valfri LLM-granskning via GitHub Models/Azure
+-- auto_learn.py        # Lar fran LLM-diff till lokala korrektioner
+-- modifiers.py         # Kanonisk modifier-namngivning (ctrl/shift/alt/windows)
+-- convert_model.py     # CLI-verktyg for modellkonvertering (KBLab -> CTranslate2)
+-- make_icon.py         # Genererar assets/icon.ico via Pillow
+-- build.bat            # PyInstaller bygge (--onedir, CUDA, VAD-assets)
+-- run.bat              # Dev-korning med beroendeinstallation
+-- requirements.txt     # Python dependencies (torch/nemo installeras separat)
```

---

## Konfiguration

Sparas i `~/.freewispr-swedish-parakeet/config.json`:

```json
{
  "hotkey": "ctrl+space",
  "model_size": "small",
  "use_cuda": true,
  "backend": "auto",
  "mic_device": null
}
```

| Nyckel | Typ | Standard | Beskrivning |
|--------|-----|----------|-------------|
| `hotkey` | string | `"ctrl+space"` | Tangentkombination for diktering |
| `model_size` | string | `"small"` | Whisper-modell: tiny/base/small/medium/large |
| `use_cuda` | bool | `true` | Anvand GPU om tillganglig |
| `backend` | string | `"auto"` | STT-backend: auto/parakeet/whisper |
| `mic_device` | string/null | `null` | Mikrofonnamn, eller `null` for auto |
| `llm_enabled` | bool | `false` | Skicka transkriberad text till LLM for granskning |
| `llm_model` | string | `"gpt-4.1-nano"` | Modell for LLM-granskning |
| `style` | string | `"casual"` | LLM-stilpreset: casual/formal/code/email/custom |
| `custom_style_prompt` | string | `""` | Egen LLM-prompt (anvands nar style=custom) |
| `paste_strategy` | string | `"auto"` | Inklistringsmetod: auto/clipboard/inject |
| `paste_threshold` | int | `200` | Teckengrns for hybrid-paste (auto-lage) |
| `min_rms` | float | `0.003` | Lagsta RMS-niva for tystnadsdetektion |
| `preroll_enabled` | bool | `false` | Hall mikrofonen aktiv for snabbare start |
| `preroll_seconds` | float | `0.5` | Langd pa pre-roll-buffert i sekunder |

LLM API-nyckeln sparas i Windows Credential Manager via `keyring` och skrivs inte till `config.json`.

### Ovriga datafiler

| Fil | Beskrivning |
|-----|-------------|
| `~/.freewispr-swedish-parakeet/corrections.json` | Personliga ordrattningar (wrong -> right) |
| `~/.freewispr-swedish-parakeet/snippets.json` | Snippets (trigger -> expansion) |
| `~/.freewispr-swedish-parakeet/learned.json` | Auto-inlarda korrigeringar fran LLM-diff |
| `~/.freewispr-swedish-parakeet/hotwords.txt` | Egna termer for Whisper (valfri) |
| `~/.freewispr-swedish-parakeet/freewispr.log` | Logfil for felsokning |
| `~/.freewispr-swedish-parakeet/models/` | Nedladdade och konverterade modeller |

---

## Decoder-optimeringar (Whisper-backend)

Foljande Whisper-parametrar anvands for basta svenska transkribering:

| Parameter | Varde | Effekt |
|-----------|-------|--------|
| `beam_size` | 1 | Greedy decoding (snabbast for diktering) |
| `repetition_penalty` | 1.1 | Mild straff pa upprepade tokens |
| `no_repeat_ngram_size` | 3 | Forbjuder exakt upprepade 3-ordskombinationer |
| `initial_prompt` | Svenska dikteringsprompt | Forankrar decodern i ratt sprak/stil |
| `hotwords` | Fran ordlista + hotwords.txt | Bias mot anvandares egna termer |
| `vad_filter` | `True` (med fallback) | Filtrerar tystnad fore transkribering |

---

## Uppdatera fran originalet

Lagg till upstream remote for att hamta forbattringar fran originalprojektet:

```bash
git remote add upstream https://github.com/x26prakhar/freewispr.git
git fetch upstream
git merge upstream/master
```

---

## Licens

[MIT](LICENSE)

---

## Tack till

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) -- Parakeet ASR-modeller
- [KBLab](https://huggingface.co/KBLab) -- Kungliga bibliotekets svenska Whisper-modeller
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- effektiv Whisper-inferens via CTranslate2
- [OpenAI Whisper](https://github.com/openai/whisper) -- den ursprungliga speech recognition-modellen
- [freewispr](https://github.com/x26prakhar/freewispr) -- originalprojektet
