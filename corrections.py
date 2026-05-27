"""
Personal dictionary — word corrections applied after transcription.
Stored at ~/.{APP_NAME}/corrections.json as {"wrong": "right", ...}
"""
import re
from pathlib import Path

from config import APP_NAME
from json_store import JsonCache

_store = JsonCache(Path.home() / f".{APP_NAME}" / "corrections.json")
_FILE = _store._path  # exposed for tests that reference corrections._FILE


def mtime() -> float:
    return _store.mtime()


def load() -> dict[str, str]:
    return _store.load()


def save(corrections: dict[str, str]):
    _store.save(corrections)


# Compiled-pattern cache for apply(). A *single* alternation regex with a
# dispatch dict — one linear scan over the text replaces all corrections,
# instead of N sequential substring scans (one per pair). For dictionaries
# with 50+ entries this is the difference between O(N*M) and O(M) per call.
_apply_master: re.Pattern[str] | None = None
_apply_lookup: dict[str, str] = {}
_apply_cache_mtime: float = -1.0


def _build_apply_cache(corr: dict[str, str]) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    if not corr:
        return None, {}
    # Sort longest first so multi-word keys match before their prefixes.
    keys = sorted(corr.keys(), key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in keys)
    pattern = re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)
    lookup = {k.lower(): v for k, v in corr.items()}
    return pattern, lookup


def _mirror_case(source: str, replacement: str) -> str:
    """Adopt the case pattern of `source` into `replacement`.

    Rules:
      - source is ALL UPPERCASE -> uppercase replacement
      - source starts with uppercase letter -> capitalize replacement
      - otherwise -> leave replacement as stored

    Avoids the surprise of dictating "MÖTE" and getting "möte" pasted
    because the dictionary entry was stored lowercase.
    """
    if not source or not replacement:
        return replacement
    # All-caps detection ignores any non-letter chars so e.g. "USA:S" still triggers.
    letters = [c for c in source if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply(text: str) -> str:
    """Replace all correction pairs (case-insensitive match, case-mirrored replacement)."""
    global _apply_master, _apply_lookup, _apply_cache_mtime
    corr = load()
    current_mt = _store._cache_mtime
    if current_mt != _apply_cache_mtime:
        _apply_master, _apply_lookup = _build_apply_cache(corr)
        _apply_cache_mtime = current_mt
    if _apply_master is None:
        return text

    def _sub(m: re.Match[str]) -> str:
        matched = m.group(0)
        replacement = _apply_lookup[matched.lower()]
        return _mirror_case(matched, replacement)

    return _apply_master.sub(_sub, text)
