"""
Hybrid text injection: keyboard.write for short texts, clipboard for long.

Two injection paths:

1. ``keyboard.write`` — types each char as a synthetic keystroke. No
   clipboard touch, no Ctrl+V grace period. ~30 ms for a one-line phrase.
   Loses Unicode chars that aren't on the current layout (rare for sv-SE
   but matters for emoji / CJK / smart-quotes).

2. ``_paste_and_restore`` (clipboard + Ctrl+V) — needed for long text
   (typing 500 chars at 5 ms each is 2.5 s of visible keystrokes) and
   for guaranteed Unicode fidelity. Pays clipboard-save/restore cost.

The threshold is configurable (``paste_threshold``). 200 chars is the
default — covers ~30 words, which is a long dictated sentence but still
keyboard-fast (~1 s of synthesised typing).

Strategy override (``paste_strategy``):
- ``"auto"`` (default) — hybrid as described above.
- ``"clipboard"`` — always paste via clipboard.
- ``"inject"``    — always type via keyboard.write.

Clipboard restore uses **polling** rather than a fixed sleep: it waits
until the target app has consumed the dictated text from the clipboard
(or until a 1.5 s timeout). This fixes the Word/RDP/Teams race where the
old 150 ms grace stole the paste mid-flight.

A monotonically increasing generation counter cancels any in-flight
restore when a new dictation starts. Prevents a stale restore from a
slow target app clobbering the clipboard the user just set themselves.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import keyboard
import pyperclip

from modifiers import CANONICAL_MODIFIERS, normalize_all
from paste import _active_window_class, _paste_shortcut

log = logging.getLogger("freewispr")

# Polling parameters for clipboard restore.
_RESTORE_POLL_INTERVAL_SEC = 0.020   # 20 ms between checks
_RESTORE_MAX_WAIT_SEC = 1.5          # give up after 1.5 s
_RESTORE_RETRY_COUNT = 3             # set clipboard up to 3x on failure
_RESTORE_RETRY_DELAY_SEC = 0.05

# Default threshold: text shorter than this is injected via keyboard.write.
# 200 chars ≈ 30 words ≈ 1 s of synthesised typing. Anything longer goes
# via the clipboard for speed.
DEFAULT_PASTE_THRESHOLD = 200

# Serialises clipboard + keyboard.send so two queued dictations can't
# interleave their paste sequences.
_INJECT_LOCK = threading.Lock()

# Generation counter — bumped on every paste so a slow restore from a
# previous paste can detect it's stale and abort.
_generation = 0
_generation_lock = threading.Lock()


def _next_generation() -> int:
    global _generation
    with _generation_lock:
        _generation += 1
        return _generation


def _current_generation() -> int:
    with _generation_lock:
        return _generation


def _release_modifiers(active_modifiers: tuple[str, ...] = ()) -> None:
    candidates = (
        normalize_all(active_modifiers) if active_modifiers else CANONICAL_MODIFIERS
    )
    for key in candidates:
        try:
            if keyboard.is_pressed(key):
                keyboard.release(key)
        except Exception:
            pass


def _inject_via_keyboard(text: str) -> bool:
    """Type ``text`` via keyboard.write. Returns True on success."""
    try:
        # delay=0 = type as fast as possible (~5 ms per char on Windows).
        # exact=False allows fallback to alt-codes for chars missing from
        # the current layout; on sv-SE this almost never triggers.
        keyboard.write(text, delay=0)
        return True
    except Exception as e:
        log.warning("keyboard.write fel: %s", e)
        return False


def _wait_for_clipboard_change(expected_marker: str, gen: int) -> None:
    """Poll until the clipboard no longer holds the dictated text, OR a
    later generation has started, OR the timeout fires.

    The target app *consumes* the clipboard immediately when it processes
    Ctrl+V — most apps just read it once and we don't actually need to
    wait. But Word/RDP/Teams sometimes paste asynchronously; the old
    150 ms fixed sleep was both too short for them and wastefully long
    for everyone else. Polling adapts to whichever app received the paste.
    """
    deadline = time.monotonic() + _RESTORE_MAX_WAIT_SEC
    while time.monotonic() < deadline:
        if _current_generation() != gen:
            # A new dictation started — don't restore, the next paste will
            # set the clipboard to its own dictated text shortly.
            log.debug("Restore: skipped (newer generation took over)")
            return
        try:
            current = pyperclip.paste()
        except Exception:
            current = ""
        if current != expected_marker:
            # Either the app consumed our text, or the user copied
            # something themselves. Either way, no point holding it.
            return
        time.sleep(_RESTORE_POLL_INTERVAL_SEC)


def _restore_clipboard(old: str, dictated_marker: str, gen: int) -> None:
    """Restore the prior clipboard content, polling-based + cancellable."""
    _wait_for_clipboard_change(dictated_marker, gen)
    # Generation check again right before write — minimises the race
    # where a brand-new paste copies its text, then ours arrives last.
    if _current_generation() != gen:
        return
    for attempt in range(1, _RESTORE_RETRY_COUNT + 1):
        try:
            pyperclip.copy(old)
            return
        except Exception as e:
            if attempt == _RESTORE_RETRY_COUNT:
                log.warning("Kunde inte aterstalla urklipp: %s", e)
                return
            time.sleep(_RESTORE_RETRY_DELAY_SEC)


def _inject_via_clipboard(text: str, gen: int) -> bool:
    """Save old clipboard, paste new text, restore old (polling-based)."""
    try:
        old = pyperclip.paste()
    except Exception:
        old = ""

    dictated_with_trailing_space = text + " "
    try:
        pyperclip.copy(dictated_with_trailing_space)
        keyboard.send(_paste_shortcut())
    except Exception as e:
        log.warning("Kunde inte skicka paste-genväg: %s", e)
        return False

    # Restore in background — the dictation worker doesn't need to wait
    # for the target app to consume the clipboard.
    threading.Thread(
        target=_restore_clipboard,
        args=(old, dictated_with_trailing_space, gen),
        daemon=True,
        name="clipboard-restore",
    ).start()
    return True


def inject(
    text: str,
    active_modifiers: tuple[str, ...] = (),
    strategy: str = "auto",
    paste_threshold: int = DEFAULT_PASTE_THRESHOLD,
) -> bool:
    """Inject ``text`` at the current cursor position.

    Args:
        text: text to paste.
        active_modifiers: modifier keys from the dictation hotkey that
            should be released before injection (avoids Start-menu opening
            when Win is part of the hotkey).
        strategy: ``"auto"`` (hybrid), ``"clipboard"``, or ``"inject"``.
        paste_threshold: in ``auto`` mode, texts longer than this fall
            back to the clipboard path. Ignored for forced strategies.

    Returns ``True`` on success, ``False`` if injection failed.
    """
    text = text.strip()
    if not text:
        return False

    _release_modifiers(active_modifiers)

    # Bump generation so any older restore in flight aborts itself.
    gen = _next_generation()

    use_keyboard = (
        strategy == "inject"
        or (strategy == "auto" and len(text) <= paste_threshold)
    )

    with _INJECT_LOCK:
        if use_keyboard:
            if _inject_via_keyboard(text + " "):
                return True
            # Fallback to clipboard if keyboard.write blew up — e.g. some
            # exotic Unicode the layout can't represent. Still better than
            # losing the dictation entirely.
            log.info("keyboard.write fallback -> clipboard")
        return _inject_via_clipboard(text, gen)


# Backwards-compatible alias: paste.py:paste_text used to be the single
# entry point. Old callers continue to work; new ones can call inject()
# directly with custom strategy.
def paste_text(
    text: str,
    active_modifiers: tuple[str, ...] = (),
    strategy: str = "auto",
    paste_threshold: int = DEFAULT_PASTE_THRESHOLD,
) -> bool:
    return inject(text, active_modifiers, strategy, paste_threshold)
