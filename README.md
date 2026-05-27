# freewispr-swedish

**Svensk speech-to-text diktering for Windows.**
Diktera var som helst. Lokal Whisper-diktering, med valfri onlinelagranskning via LLM.

> **Fork av:** [x26prakhar/freewispr](https://github.com/x26prakhar/freewispr)
> **Licens:** MIT

---

## Vad ar freewispr-swedish?

freewispr-swedish ar en modifierad version av freewispr optimerad for **svenska**. Istallet for OpenAI:s Whisper-modeller anvands **KBLab:s svenska Whisper-modeller** som ger upp till **47% lagre WER** (Word Error Rate) pa svenska.

Standardmodellen ar `KBLab/kb-whisper-small` -- battre precision pa svenska an OpenAI:s `whisper-small`.

---

## Funktionalitet

- **Diktering** -- hall tangent nedtryckt, prata, slapp. Texten klistras in direkt vid markoren
- **Flytande indikator** -- visar Lyssnar / Transkriberar / Klar / Fel
- **Ljudatergivning** -- mjuka pop-ljud vid inspelningsstart och -stopp
- **Mikrofonval** -- valj mikrofon i installningar (WASAPI, DirectSound, MME)
- **Tystnadsdetektion** -- avvisar for tysta inspelningar automatiskt (RMS-baserad)
- **Personlig ordlista** -- lagg till rattningar for ord som transkriberas fel
- **Hotwords** -- mata in egna termer/namn som Whisper ska prioritera
- **Snippets** -- textmallar som expanderas automatiskt
- **Flermodell-stod** -- tiny, base, small, medium, large (alla KBLab)
- **Starta med Windows** -- enkel toggle i menyn
- **Systemfack** -- lever diskret i bakgrunden
- **Offline-lage** -- Whisper-diktering sker lokalt efter forsta modellnedladdningen
- **Valfri LLM-granskning** -- kan skicka transkriberad text till GitHub Models/Azure for extra textkorrigering
- **GPU-stod** -- automatisk CUDA-detektion for NVIDIA-grafikkort

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

Som standard transkriberas ljud lokalt med KBLab Whisper. Appen kan kontakta natverk i foljande fall:

- Forsta modellstarten laddar ner vald KBLab-modell fran Hugging Face om modellen saknas lokalt.
- LLM-granskning ar avstangd som standard, men nar du aktiverar den skickas transkriberad text till GitHub Models/Azure for korrigering.
- LLM-granskning skickar texten efter lokal Whisper-transkribering, inte ljudfilen.
- Dikterad text loggas inte som standard; loggen innehaller metadata som langd, modell, latency och status.
- Text klistras in via systemets urklipp och appen forsoker aterstalla tidigare urklippsinnehall efter paste.
- Om LLM-granskning andrar text kan lokala ordkorrigeringar sparas i `learned.json` och `corrections.json` for att forbättra framtida lokal transkribering.

---

## Bygga fran kallkod

**Krav:** Python 3.10+, Windows 10/11

```bash
# Klona repot
git clone https://github.com/hhammarstrand/freewispr-swedish.git
cd freewispr-swedish

# Installera dependencies
pip install -r requirements.txt

# (Valfritt) Installera GPU-stod (NVIDIA, ~2.5 GB nedladdning)
pip install torch --index-url https://download.pytorch.org/whl/cu124

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

## Modeller

freewispr-swedish anvander [KBLab:s Whisper-modeller](https://huggingface.co/KBLab) tranade pa over 50 000 timmar svenskt tal.

| Modell | WER (svenska) | Storlek | Jamforelse OpenAI |
|--------|---------------|---------|-------------------|
| `tiny` | 13.2 | ~40 MB | 59.2 |
| `base` | 9.1 | ~150 MB | 39.6 |
| **`small`** | **7.3** | **~500 MB** | **20.6** |
| `medium` | 6.6 | ~1.5 GB | 15.8 |
| `large` | -- | ~3 GB | -- |

**Standard:** `small` -- basta balansen mellan hastighet och precision for svenska.

Modeller sparas i `~/.freewispr-swedish/models/` och laddas ner automatiskt vid forsta anvandning.

### Konvertera modeller (medium/large)

Medium- och large-modellerna kan behova konverteras till CTranslate2-format for att undvika vocabulary-krascher:

```bash
pip install ctranslate2 transformers
python convert_model.py medium
python convert_model.py large
```

Konverterade modeller sparas i `~/.freewispr-swedish/models/kb-whisper-{size}-ct2/`.

---

## Installningar

Hogerklicka pa systemfacksikonen och valj **Installningar**.

- **Snabbtangent** -- klicka och tryck valfri tangentkombination
- **Mikrofon** -- valj specifik mikrofon eller "Auto"
- **Modell** -- valj storlek (tiny/base/small/medium/large)
- **GPU (CUDA)** -- sla pa/av GPU-acceleration

---

## Hotwords (personlig ordlista for Whisper)

Hotwords ar termer, namn och fraser som Whisper ska prioritera vid transkribering. De forbattrar precision for ovanliga ord, egennamn och facktermer.

Hotwords hamtas fran tva kallor:

1. **Personlig ordlista** -- de korrekta varden du lagt in via systemfacket ("Personlig ordlista")
2. **hotwords.txt** -- valfri fil pa `~/.freewispr-swedish/hotwords.txt`, ett ord/fras per rad

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

Inspelning sker i mikrofonens nativa samplerate (t.ex. 48 kHz) och resamplas till 16 kHz for Whisper med `scipy.signal.resample_poly` (anti-alias FIR-filter). Flerkanaliga mikrofoner mixas till mono automatiskt.

---

## Teknisk arkitektur

```
freewispr-swedish/
+-- main.py            # Entry point: systemfack, threading, applifecycle
+-- transcriber.py     # KB-Whisper + CUDA + decoder-optimeringar + hotwords
+-- dictation.py       # Dikteringslogik: tangent -> spela in -> transkribera -> klistra
+-- audio.py           # Mikrofoninspelning (WASAPI prio, resample, enhetsval)
+-- paste.py           # Urklipp via pyperclip + keyboard.send (modifier pre-release)
+-- sounds.py          # Syntetiserade pop-ljud for inspelningsatergivning
+-- ui.py              # Tkinter: flytande indikator, installningar, snippets, ordlista
+-- config.py          # JSON konfiguration (~/.freewispr-swedish/config.json)
+-- corrections.py     # Personliga ordrattningar (~/.freewispr-swedish/corrections.json)
+-- snippets.py        # Textmallar/expansion (~/.freewispr-swedish/snippets.json)
+-- convert_model.py   # CLI-verktyg for modellkonvertering (KBLab -> CTranslate2)
+-- make_icon.py       # Genererar assets/icon.ico via Pillow
+-- build.bat          # PyInstaller bygge (--onedir, CUDA, VAD-assets)
+-- run.bat            # Dev-korning med beroendeinstallation
+-- requirements.txt   # Python dependencies (torch installeras separat)
```

---

## Konfiguration

Sparas i `~/.freewispr-swedish/config.json`:

```json
{
  "hotkey": "ctrl+space",
  "model_size": "small",
  "use_cuda": true,
  "mic_device": null
}
```

| Nyckel | Typ | Standard | Beskrivning |
|--------|-----|----------|-------------|
| `hotkey` | string | `"ctrl+space"` | Tangentkombination for diktering |
| `model_size` | string | `"small"` | Whisper-modell: tiny/base/small/medium/large |
| `use_cuda` | bool | `true` | Anvand GPU om tillganglig |
| `mic_device` | string/null | `null` | Mikrofonnamn, eller `null` for auto |
| `llm_enabled` | bool | `false` | Skicka transkriberad text till LLM for granskning |
| `llm_model` | string | `"gpt-4.1-nano"` | Modell for LLM-granskning |

LLM API-nyckeln sparas i Windows Credential Manager via `keyring` och skrivs inte till `config.json`.

### Ovriga datafiler

| Fil | Beskrivning |
|-----|-------------|
| `~/.freewispr-swedish/corrections.json` | Personliga ordrattningar (wrong -> right) |
| `~/.freewispr-swedish/snippets.json` | Snippets (trigger -> expansion) |
| `~/.freewispr-swedish/hotwords.txt` | Egna termer for Whisper (valfri) |
| `~/.freewispr-swedish/freewispr.log` | Logfil for felsokning |
| `~/.freewispr-swedish/models/` | Nedladdade och konverterade modeller |

---

## Decoder-optimeringar

Foljande Whisper-parametrar anvands for basta svenska transkribering:

| Parameter | Varde | Effekt |
|-----------|-------|--------|
| `beam_size` | 5 | Standard beam search (battre an greedy) |
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

- [KBLab](https://huggingface.co/KBLab) -- Kungliga bibliotekets svenska Whisper-modeller
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- effektiv Whisper-inferens via CTranslate2
- [OpenAI Whisper](https://github.com/openai/whisper) -- den ursprungliga speech recognition-modellen
- [freewispr](https://github.com/x26prakhar/freewispr) -- originalprojektet
