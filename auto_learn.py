"""
Auto-learning — track LLM corrections and promote frequent ones
to the personal corrections dictionary.

How it works:
  1. Each time the LLM changes a word, record_correction() is called
     with the before/after text.
  2. Word-level diffs are extracted (single word changes only — safe).
  3. Each diff is tallied in a frequency file (~/.freewispr-swedish/learned.json).
  4. When a correction reaches PROMOTE_THRESHOLD occurrences, it is
     automatically added to corrections.json AND as a hotword.
  5. This means Whisper itself gets better (hotword) AND the local
     post-processing catches the error (correction) — without needing
     the LLM next time.

The learned.json format:
  {
    "motte": {"correct": "möte", "count": 5, "promoted": true},
    "pratchar": {"correct": "Prakhar", "count": 2, "promoted": false},
    ...
  }
"""
import logging
from difflib import SequenceMatcher
from pathlib import Path

import corrections as corr_module
from json_store import load_json, save_json_atomic
from config import APP_NAME

log = logging.getLogger("freewispr")

LEARNED_FILE = Path.home() / f".{APP_NAME}" / "learned.json"

# How many times the same correction must occur before auto-promoting
PROMOTE_THRESHOLD = 3


def _load_learned() -> dict:
    """Load the learned corrections tally."""
    if LEARNED_FILE.exists():
        return load_json(LEARNED_FILE, {})
    return {}


def _save_learned(data: dict) -> None:
    """Save the learned corrections tally."""
    LEARNED_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_json_atomic(LEARNED_FILE, data)
    except Exception as e:
        log.warning("Kunde inte spara learned.json: %s", e)


def _extract_word_diffs(before: str, after: str) -> list[tuple[str, str]]:
    """Extract single-word replacements between before and after text.

    Uses ``difflib.SequenceMatcher`` opcodes so we still recover word
    swaps when the LLM also added or removed words elsewhere in the
    sentence (the previous same-length zip check rejected those entirely).

    Only ``replace`` opcodes that swap exactly one word for one word are
    learned — multi-word rewrites are too noisy to promote automatically.

    Returns list of (wrong_word, correct_word) tuples.
    """
    before_words = before.split()
    after_words = after.split()
    if not before_words or not after_words:
        return []

    diffs: list[tuple[str, str]] = []
    matcher = SequenceMatcher(a=before_words, b=after_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        # Only single-word -> single-word swaps are safe to learn.
        if (i2 - i1) != 1 or (j2 - j1) != 1:
            continue
        bw = before_words[i1].strip(".,;:!?\"'()[]")
        aw = after_words[j1].strip(".,;:!?\"'()[]")
        if not (bw and aw):
            continue
        if bw.lower() == aw.lower():
            continue
        # Skip very short words — high false-positive rate.
        if len(bw) < 2 or len(aw) < 2:
            continue
        diffs.append((bw.lower(), aw))

    return diffs


def record_correction(before: str, after: str) -> None:
    """Record a LLM correction and auto-promote if threshold reached.

    Called by transcriber.py each time the LLM changes the text.
    """
    diffs = _extract_word_diffs(before, after)
    if not diffs:
        return

    learned = _load_learned()
    promoted_any = False
    recorded: list[tuple[str, int]] = []

    for wrong, correct in diffs:
        entry = learned.get(wrong)
        if entry and entry.get("promoted"):
            # Already promoted — skip
            continue

        if entry:
            entry["count"] = entry.get("count", 0) + 1
            # Update correct form (use latest — LLM may improve capitalization)
            entry["correct"] = correct
        else:
            entry = {"correct": correct, "count": 1, "promoted": False}
            learned[wrong] = entry

        recorded.append((wrong, entry["count"]))

        # Promote if threshold reached
        if entry["count"] >= PROMOTE_THRESHOLD and not entry["promoted"]:
            _promote(wrong, correct)
            entry["promoted"] = True
            promoted_any = True

    if recorded:
        # One aggregated log line instead of one per diff — keeps the log
        # readable when the LLM rewrites a long sentence with many fixes.
        summary = ", ".join(f"{w}={c}/{PROMOTE_THRESHOLD}" for w, c in recorded)
        log.info("Auto-lärning: registrerade %d korrigering(ar) [%s]",
                 len(recorded), summary)

    _save_learned(learned)

    if promoted_any:
        log.info("Auto-lärning befordrade nya korrigeringar till ordlistan")


def _promote(wrong: str, correct: str) -> None:
    """Add a learned correction to the personal dictionary.

    This has two effects:
      1. corrections.json — future Whisper output is corrected locally
      2. hotwords — the correct word becomes a hotword for Whisper's decoder
         (via _load_hotwords() in transcriber.py which reads corrections)
    """
    corrs = corr_module.load()
    if wrong not in corrs:
        corrs[wrong] = correct
        corr_module.save(corrs)
        log.info("Befordrad till ordlistan")
    else:
        log.debug("Korrigering finns redan i ordlistan")


def get_stats() -> dict:
    """Return learning statistics for UI display."""
    learned = _load_learned()
    total = len(learned)
    promoted = sum(1 for e in learned.values() if e.get("promoted"))
    pending = total - promoted
    return {
        "total": total,
        "promoted": promoted,
        "pending": pending,
        "entries": learned,
    }
