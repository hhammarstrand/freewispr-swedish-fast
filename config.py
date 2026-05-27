from pathlib import Path

from json_store import load_json, save_json_atomic

try:
    import keyring
except Exception:
    keyring = None

APP_NAME = "freewispr-swedish-parakeet"
CONFIG_DIR = Path.home() / f".{APP_NAME}"
CONFIG_FILE = CONFIG_DIR / "config.json"
_KEYRING_SERVICE = APP_NAME
_KEYRING_USERNAME = "llm_api_key"

DEFAULTS = {
    "hotkey": "ctrl+space",
    "model_size": "small",     # tiny/base/small/medium/large (whisper fallback size)
    "use_cuda": True,         # True = auto-detect GPU, False = force CPU
    "backend": "auto",        # "auto" (parakeet if available, else whisper), "parakeet", "whisper"
    "mic_device": None,       # None = auto-detect, or device name string
    "llm_enabled": False,     # LLM post-processing of transcribed text
    "llm_api_key": "",        # Runtime only; saved in Windows Credential Manager when available
    "llm_model": "gpt-4.1-nano",  # Which LLM model to use
    "llm_privacy_accepted": False,
    # Lägsta RMS-nivå (0.0-1.0) som räknas som tal. Se DEFAULT_MIN_RMS i dictation.py.
    "min_rms": 0.003,
    # Paste pipeline — "auto" (hybrid), "clipboard", or "inject".
    # Hybrid uses keyboard.write for text <= paste_threshold chars and
    # clipboard + Ctrl+V for anything longer. ~150 ms latency win on
    # short utterances at the cost of slightly slower long ones.
    "paste_strategy": "auto",
    "paste_threshold": 200,
    # Style preset for LLM polish. One of casual/formal/code/email/custom.
    # When "custom", the value of custom_style_prompt is appended to the
    # base system prompt instead of a built-in preset.
    "style": "casual",
    "custom_style_prompt": "",
    # Pre-roll: keep a ~500 ms rolling buffer of mic input always armed so
    # the first ~half second of speech after pressing the hotkey isn't
    # clipped. Opt-in: when ON, the mic LED stays lit whenever the app is
    # running. Off by default for privacy.
    "preroll_enabled": False,
    "preroll_seconds": 0.5,
}


def _get_secret() -> str:
    if not keyring:
        return ""
    try:
        return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME) or ""
    except Exception:
        return ""


def _set_secret(value: str) -> bool:
    if not keyring:
        return False
    try:
        if value:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, value)
        else:
            keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        return True
    except Exception:
        return False


def can_store_secret() -> bool:
    if not keyring:
        return False
    test_user = f"{_KEYRING_USERNAME}_probe"
    test_value = "probe"
    try:
        keyring.set_password(_KEYRING_SERVICE, test_user, test_value)
        ok = keyring.get_password(_KEYRING_SERVICE, test_user) == test_value
        try:
            keyring.delete_password(_KEYRING_SERVICE, test_user)
        except Exception:
            pass
        return ok
    except Exception:
        return False


def load():
    CONFIG_DIR.mkdir(exist_ok=True)
    if CONFIG_FILE.exists():
        data = load_json(CONFIG_FILE, {})
        cfg = {**DEFAULTS, **data}
    else:
        data = {}
        cfg = DEFAULTS.copy()
    legacy_api_key = cfg.pop("llm_api_key", "")
    data.pop("llm_api_key", None)
    stored_api_key = _get_secret()
    if legacy_api_key and not stored_api_key:
        if _set_secret(legacy_api_key):
            stored_api_key = legacy_api_key
            save_json_atomic(CONFIG_FILE, data)
        else:
            # Keyring may be unavailable in a broken/frozen environment. Do
            # not delete the user's only copy of the token unless migration
            # succeeds; keep runtime behavior unchanged for this launch.
            stored_api_key = legacy_api_key
    cfg["llm_api_key"] = stored_api_key
    return cfg


def save(cfg):
    CONFIG_DIR.mkdir(exist_ok=True)
    data = cfg.copy()
    api_key = data.pop("llm_api_key", "")
    old_secret = _get_secret()
    try:
        if api_key:
            if not _set_secret(api_key):
                raise RuntimeError("Kunde inte spara API-nyckeln i Windows Credential Manager")
        elif old_secret and not _set_secret(""):
            raise RuntimeError("Kunde inte ta bort API-nyckeln från Windows Credential Manager")
        save_json_atomic(CONFIG_FILE, data)
    except Exception:
        # Keep Credential Manager consistent with the JSON config on disk.
        if old_secret:
            _set_secret(old_secret)
        elif api_key:
            _set_secret("")
        raise
