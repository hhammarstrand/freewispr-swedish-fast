import logging
import re
import threading
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

import corrections as corr_module
from config import APP_NAME

log = logging.getLogger("freewispr")

CONFIG_DIR = Path.home() / f".{APP_NAME}"
MODEL_DIR = CONFIG_DIR / "models"
HOTWORDS_FILE = CONFIG_DIR / "hotwords.txt"


def _text_meta(text: str) -> str:
    words = len(text.split())
    return f"chars={len(text)}, words={words}"


def _find_local_model(repo_name: str) -> str | None:
    """Return local snapshot path if the model is already downloaded.

    Checks two locations in order:
      1. Manually converted CTranslate2 model:
         MODEL_DIR/kb-whisper-{size}-ct2/model.bin
         These are converted via ctranslate2.converters.TransformersConverter
         and are known to work correctly (no vocabulary mismatch issues).
      2. HuggingFace snapshot:
         MODEL_DIR/models--<org>--<name>/snapshots/<hash>/model.bin
         These may need vocabulary patching for large/medium models.
    """
    # 1. Check for manually converted ct2 model first
    # repo_name is e.g. "KBLab/kb-whisper-large" → extract "kb-whisper-large"
    short_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
    ct2_dir = MODEL_DIR / f"{short_name}-ct2"
    if ct2_dir.exists() and (ct2_dir / "model.bin").exists():
        log.info("Hittade konverterad ct2-modell: %s", ct2_dir)
        return str(ct2_dir)

    # 2. Fall back to HuggingFace snapshot
    safe_name = repo_name.replace("/", "--")
    model_dir = MODEL_DIR / f"models--{safe_name}"
    if not model_dir.exists():
        return None
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    # Pick the newest snapshot that contains a model.bin
    for snap in sorted(snapshots.iterdir(), reverse=True):
        if (snap / "model.bin").exists():
            _patch_vocabulary(snap)
            return str(snap)
    return None


def _patch_vocabulary(snapshot_dir: Path) -> None:
    """Fix KBLab large model vocabulary mismatch.

    KBLab/kb-whisper-large has 51866 tokens (extra <|30.00|> timestamp)
    while CTranslate2 expects exactly 51865. This causes:
      RuntimeError: [json.exception.type_error.305] cannot use operator[]
      with a string argument with null
    We trim the extra token on disk once.

    SAFETY: Mutating the HuggingFace cache in place is fragile — a `huggingface-cli
    download` (or any cache validation) will redownload the original and undo us.
    To make the patch traceable and recoverable we:
      1. Save the original to ``vocabulary.json.orig`` before overwriting.
      2. Drop a ``.freewispr-patched`` marker so we don't repeatedly log/patch.
      3. Loudly recommend the user run ``python convert_model.py large`` for a
         proper fix (writes a clean ct2 model next to the HF cache).
    """
    import json
    vocab_path = snapshot_dir / "vocabulary.json"
    marker = snapshot_dir / ".freewispr-patched"
    if not vocab_path.exists():
        return
    if marker.exists():
        return  # already patched in a previous run
    try:
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        if isinstance(vocab, list) and len(vocab) > 51865:
            log.warning(
                "vocabulary.json har %d tokens (förväntade 51865). "
                "Patchar HuggingFace-cachen på plats — kör hellre "
                "'python convert_model.py large' för en permanent lösning.",
                len(vocab),
            )
            backup = snapshot_dir / "vocabulary.json.orig"
            if not backup.exists():
                vocab_path.replace(backup)
                # vocab_path no longer exists; recreate it below.
            vocab = vocab[:51865]
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
            marker.write_text("trimmed to 51865 tokens by freewispr-swedish\n",
                              encoding="utf-8")
    except Exception as e:
        log.warning("Kunde inte patcha vocabulary.json: %s", e)

# KBLab model mapping for Swedish Whisper
KBLAB_MODELS = {
    "tiny": "KBLab/kb-whisper-tiny",
    "base": "KBLab/kb-whisper-base",
    "small": "KBLab/kb-whisper-small",
    "medium": "KBLab/kb-whisper-medium",
    "large": "KBLab/kb-whisper-large",
}

# Whisper noise/placeholder tokens to strip (always, regardless of settings).
# These appear when Whisper hallucinates on silence or background noise.
_NOISE_PLACEHOLDERS = re.compile(
    r'\[BLANK_AUDIO\]'
    r'|\[SILENCE\]'
    r'|<\|nospeech\|>'
    r'|<\|endoftext\|>'
    # Bracketed noise labels (English & Swedish)
    r'|\[(?:'
    r'applause|applåder|background noise|bakgrundsljud|blank audio'
    r'|breathing|andning|cough|hosta|exhale|inhale'
    r'|laughter|laughing|skratt|music|musik'
    r'|noise|ljud|silence|tystnad|sigh|suckar'
    r'|sniffing|static|brus|unclear speech|otydligt tal'
    r'|unintelligible|wind|vind|wind noise'
    r')\]'
    # Same with parentheses
    r'|\((?:'
    r'applause|applåder|background noise|bakgrundsljud|blank audio'
    r'|breathing|andning|cough|hosta|exhale|inhale'
    r'|laughter|laughing|skratt|music|musik'
    r'|noise|ljud|silence|tystnad|sigh|suckar'
    r'|sniffing|static|brus|unclear speech|otydligt tal'
    r'|unintelligible|wind|vind|wind noise'
    r')\)',
    re.IGNORECASE,
)

# Pre-compiled regex patterns for _postprocess (compiled once at module load).
# Pattern (2) for repeated words is potentially O(n²) on pathological input,
# but Whisper outputs rarely exceed a few hundred words; compiling once
# avoids the per-call setup cost (~5-20 ms saved on longer paragraphs).
_RE_REPEAT_WORD = re.compile(r'\b(\w+)(\s+\1){1,}\b', re.IGNORECASE | re.UNICODE)
_RE_REPEAT_PHRASE = re.compile(r'\b((?:\w+\s+){1,3}\w+)(\s+\1)+\b', re.IGNORECASE | re.UNICODE)
_RE_SPACE_BEFORE_PUNCT = re.compile(r'\s+([.,;:!?])')
_RE_SPACE_AFTER_PUNCT = re.compile(r'([.,;:!?])([A-Za-z\u00C0-\u00F6\u00F8-\u00FF])')
_RE_REPEAT_STRONG_PUNCT = re.compile(r'([.!?]){2,}')
_RE_REPEAT_SOFT_PUNCT = re.compile(r'([,;:]){2,}')
_RE_LEADING_PUNCT = re.compile(r'^[.,;:!?\s]+')
_RE_MULTISPACE = re.compile(r'\s{2,}')

# Unicode normalization via str.translate — single C-level pass instead of
# six chained str.replace calls (5-10× faster on long strings).
_UNICODE_NORMALIZE = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
})


def _postprocess(text: str) -> str:
    """Clean up Whisper output for better readability.

    Handles real issues that KBLab models produce:
    - Repeated words/phrases (Whisper stutter)
    - Whitespace before punctuation
    - Multiple punctuation in a row
    - Stray leading/trailing punctuation
    - Unicode normalization (smart quotes, dashes)
    """
    if not text:
        return text

    # 1. Normalize unicode quotes and dashes (single translation table pass)
    text = text.translate(_UNICODE_NORMALIZE)

    # 2. Remove repeated words: "det det" → "det", "jag jag jag" → "jag"
    text = _RE_REPEAT_WORD.sub(r'\1', text)

    # 3. Remove repeated short phrases (2-4 words): "det var bra det var bra" → "det var bra"
    text = _RE_REPEAT_PHRASE.sub(r'\1', text)

    # 4. Fix whitespace before punctuation: "hej , du" → "hej, du"
    text = _RE_SPACE_BEFORE_PUNCT.sub(r'\1', text)

    # 5. Ensure space after punctuation (but not digits: "3.14"): "hej.du" → "hej. du"
    text = _RE_SPACE_AFTER_PUNCT.sub(r'\1 \2', text)

    # 6. Collapse multiple punctuation: "hej..." → "hej.", "hej,," → "hej,"
    text = _RE_REPEAT_STRONG_PUNCT.sub(r'\1', text)
    text = _RE_REPEAT_SOFT_PUNCT.sub(r'\1', text)

    # 7. Strip leading punctuation
    text = _RE_LEADING_PUNCT.sub('', text)

    # 8. Collapse multiple spaces
    text = _RE_MULTISPACE.sub(' ', text).strip()

    # 9. Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text


def _check_cuda() -> bool:
    """Check if CUDA (GPU) is available. Fails fast if torch is broken."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _get_device_and_compute(use_cuda: bool) -> tuple:
    """
    Determine device and compute type based on CUDA setting.
    Returns (device, compute_type, cuda_used).
    """
    cuda_available = _check_cuda()

    if use_cuda and cuda_available:
        return ("cuda", "float16", True)
    elif use_cuda and not cuda_available:
        log.warning("CUDA begärt men ingen GPU hittades. Använder CPU.")
        return ("cpu", "int8", False)
    else:
        return ("cpu", "int8", False)


# Initial prompts guide Whisper toward the right language and style.
# This dramatically improves first-word accuracy and reduces hallucinations.
# Include a few natural Swedish phrases to anchor the decoder.
_INITIAL_PROMPTS = {
    "sv": (
        "Hej, det här är en diktering på svenska."
        " Jag dikterar text med korrekt interpunktion och stavning."
    ),
    "en": "Hello, this is a dictation in English.",
}


def _load_hotwords() -> str | None:
    """Build a comma-separated hotwords string for faster-whisper.

    Sources (combined, deduplicated):
      1. The *correct* values from the personal corrections dictionary.
         These are proper nouns, names, and terms the user cares about.
      2. An optional hotwords.txt file at ~/.freewispr-swedish/hotwords.txt
         (one word or phrase per line, blank lines and # comments ignored).

    Returns None if no hotwords are available.
    """
    words: set[str] = set()

    # 1. Correction dictionary → the "right" (target) values
    try:
        for _wrong, right in corr_module.load().items():
            term = right.strip()
            if term:
                words.add(term)
    except Exception as e:
        log.debug("Kunde inte läsa ordlista för hotwords: %s", e)

    # 2. hotwords.txt (optional)
    if HOTWORDS_FILE.exists():
        try:
            for line in HOTWORDS_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    words.add(line)
            log.info("Laddade %d hotwords från %s", len(words), HOTWORDS_FILE)
        except Exception as e:
            log.warning("Kunde inte läsa hotwords.txt: %s", e)

    if not words:
        return None

    result = ", ".join(sorted(words))
    log.debug("Hotwords (%d st): %s", len(words), result[:200])
    return result


# In-memory hotwords cache — avoids re-reading disk on every transcription.
# Invalidated when corrections.json or hotwords.txt change (checked by mtime).
_hotwords_cache: str | None = None
_hotwords_mtime: tuple[float, float] = (0.0, 0.0)  # (corrections_mtime, hotwords_mtime)


def _get_hotwords_cached() -> str | None:
    """Return cached hotwords string, reloading only if source files changed."""
    global _hotwords_cache, _hotwords_mtime

    corr_mt = corr_module.mtime()
    hw_mt = HOTWORDS_FILE.stat().st_mtime if HOTWORDS_FILE.exists() else 0.0
    current = (corr_mt, hw_mt)

    if current != _hotwords_mtime:
        _hotwords_cache = _load_hotwords()
        _hotwords_mtime = current
        log.debug("Hotwords-cache uppdaterad (mtime ändrad)")

    return _hotwords_cache


class Transcriber:
    def __init__(self, model_size: str = "small", language: str = "sv",
                 use_cuda: bool = True,
                 backend: str = "auto",
                 llm_enabled: bool = False, llm_api_key: str = "",
                 llm_model: str = "gpt-4.1-nano",
                 style: str = "casual",
                 custom_style_prompt: str = ""):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.llm_enabled = llm_enabled
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.style = style
        self.custom_style_prompt = custom_style_prompt
        self._model_lock = threading.RLock()
        self._parakeet = None
        self.model = None

        # Try Parakeet first when backend is "auto" or "parakeet" and CUDA is available.
        if backend in ("auto", "parakeet"):
            try:
                from parakeet_backend import ParakeetBackend
                from parakeet_backend import is_available as _pk_avail
                if _pk_avail():
                    log.info("Parakeet tillgänglig — laddar Parakeet-backend...")
                    self._parakeet = ParakeetBackend("cuda")
                    self.model_size = "parakeet"
                    self.backend_name = "parakeet"
                    if self.llm_enabled and self.llm_api_key:
                        log.info("LLM-granskning aktiverad: %s", self.llm_model)
                    return  # ParakeetBackend starts its own warmup thread
                elif backend == "parakeet":
                    raise RuntimeError(
                        "Parakeet begärd men NeMo är inte installerat eller CUDA saknas.\n"
                        "Installera: pip install nemo_toolkit[asr]"
                    )
            except ImportError:
                if backend == "parakeet":
                    raise
                log.debug("parakeet_backend inte tillgänglig, faller tillbaka till Whisper")
            except Exception as e:
                if backend == "parakeet":
                    raise
                log.warning("Parakeet-initiering misslyckades (%s), faller tillbaka till Whisper", e)

        # Whisper path
        model_name = KBLAB_MODELS.get(model_size, model_size)
        model_path = _find_local_model(model_name)
        device, compute_type, cuda_used = _get_device_and_compute(use_cuda)

        # Fail-closed when the model isn't cached locally: faster-whisper would
        # otherwise silently reach out to huggingface.co. That's a privacy
        # surprise for a "local dictation" tool and an OS-dependent failure for
        # offline users. Refuse and tell the user exactly how to fix it.
        if not model_path:
            raise FileNotFoundError(
                f"Whisper-modellen '{model_size}' ({model_name}) finns inte "
                f"lokalt i {MODEL_DIR}. freewispr-swedish kontaktar inte "
                f"internet automatiskt — kör först:\n"
                f"    python convert_model.py {model_size}\n"
                f"för att ladda ned och konvertera modellen."
            )

        log.info("Laddar Whisper '%s' från lokal cache (%s)...", model_size, device)
        if cuda_used:
            log.info("GPU: NVIDIA CUDA aktiverad")

        self.model_size = model_size
        self.backend_name = "whisper"
        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            download_root=str(MODEL_DIR),
            local_files_only=True,  # belt-and-braces: never hit the network
        )
        log.info("Whisper '%s' (%s) laddad OK [%s, %s]", model_size, model_name, device, compute_type)
        if self.llm_enabled and self.llm_api_key:
            log.info("LLM-granskning aktiverad: %s", self.llm_model)

        # Warm up CTranslate2 kernels / CUDA workspaces on a background thread.
        self._warmed = False
        threading.Thread(target=self._warmup, name="whisper-warmup",
                         daemon=True).start()

    def _warmup(self) -> None:
        """Run a single silent inference to pre-allocate inference state."""
        try:
            import time as _time
            t0 = _time.monotonic()
            silent = np.zeros(16000, dtype=np.float32)  # 1 s of silence
            with self._model_lock:
                model = self.model
                if model is None:
                    return
                segments, _info = model.transcribe(
                    silent,
                    language=self.language,
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                )
                # Force the lazy generator to actually run the decoder while
                # holding the lock; faster-whisper is lazy here.
                for _ in segments:
                    pass
            self._warmed = True
            log.info("Whisper-warmup klar pa %.0f ms", (_time.monotonic() - t0) * 1000)
        except Exception as e:
            log.debug("Whisper-warmup misslyckades (ignoreras): %s", e)

    def close(self) -> None:
        """Release backend model(s) to free VRAM/RAM."""
        parakeet = getattr(self, "_parakeet", None)
        if parakeet is not None:
            self._parakeet = None
            parakeet.close()
            return
        import gc
        try:
            with self._model_lock:
                model = getattr(self, "model", None)
                if model is None:
                    return
                self.model = None  # type: ignore[assignment]
                del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            log.debug("Kunde inte frigora modell rent: %s", e)

    def _transcribe_raw_whisper(self, audio: np.ndarray) -> str:
        """Raw Whisper inference — returns concatenated segment text."""
        prompt = _INITIAL_PROMPTS.get(self.language, "")
        hotwords = _get_hotwords_cached()
        raw = ""
        for use_vad in (True, False):
            try:
                with self._model_lock:
                    model = self.model
                    if model is None:
                        raise RuntimeError("Whisper-modellen är stängd")
                    segments, _info = model.transcribe(
                        audio,
                        language=self.language,
                        beam_size=1,
                        best_of=1,
                        vad_filter=use_vad,
                        vad_parameters={"min_silence_duration_ms": 500} if use_vad else None,
                        initial_prompt=prompt or None,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        hotwords=hotwords,
                    )
                    raw = " ".join(s.text.strip() for s in segments)
                break
            except RuntimeError as e:
                if use_vad:
                    log.warning("VAD-transkribering kraschade: %s — försöker utan VAD", e)
                    continue
                raise
        return raw

    def transcribe_local(self, audio: np.ndarray) -> str:
        """Transcribe using the active backend, apply corrections and post-processing."""
        peak = max(float(audio.max()), -float(audio.min())) if audio.size else 0.0
        log.info("Transkriberar: %d samples, peak=%.4f, backend=%s",
                 len(audio), peak, self.backend_name)

        if self._parakeet is not None:
            raw = self._parakeet.transcribe_raw(audio)
        else:
            raw = self._transcribe_raw_whisper(audio)

        log.info("Rå text mottagen (%s)", _text_meta(raw))
        text = _NOISE_PLACEHOLDERS.sub("", raw)
        text = corr_module.apply(text)
        text = _postprocess(text)
        log.info("Resultat (lokal) klart (%s)", _text_meta(text))
        return text

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio and apply LLM polish synchronously (backward compat)."""
        text = self.transcribe_local(audio)

        # LLM polishing — optional, never blocks on failure
        if self.llm_enabled and self.llm_api_key and text:
            from auto_learn import record_correction
            from llm_polish import polish

            result = polish(text, self.llm_api_key, self.llm_model,
                            style=self.style,
                            custom_prompt=self.custom_style_prompt)
            if result.changed:
                # Feed the before/after to auto-learning
                record_correction(text, result.text)
                text = result.text
                log.info("Resultat (LLM) klart (%dms, %s)",
                         result.latency_ms, _text_meta(text))

        return text
