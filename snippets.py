"""
Snippet library — trigger words that expand to longer phrases.
Stored at ~/.{APP_NAME}/snippets.json as {"trigger": "expansion", ...}
"""
from pathlib import Path

from config import APP_NAME
from json_store import JsonCache

_store = JsonCache(Path.home() / f".{APP_NAME}" / "snippets.json")
_FILE = _store._path  # exposed for tests that reference snippets._FILE


def load() -> dict[str, str]:
    return _store.load()


def save(snippets: dict[str, str]):
    _store.save(snippets)


def expand(text: str) -> str:
    """
    If the full transcribed text (stripped, lowercased, punctuation removed)
    exactly matches a snippet trigger, return the expansion. Otherwise
    return text unchanged.

    Whisper often appends `.` / `?` / `!` to short utterances; stripping
    them before lookup means "hälsning." still triggers a "hälsning"
    snippet.
    """
    snips = load()
    if not snips:
        return text
    key = text.strip().lower().rstrip(".,!?:;…")
    return snips.get(key, text)
