import ctypes
import logging
import threading
import time

import keyboard
import pyperclip

from modifiers import CANONICAL_MODIFIERS, normalize_all

log = logging.getLogger("freewispr")

_RESTORE_ATTEMPTS = 3
_RESTORE_DELAY_SEC = 0.05
_RESTORE_GRACE_SEC = 0.15
_PASTE_LOCK = threading.Lock()


def _active_window_class() -> str:
    """Best-effort Win32 class name for the foreground window."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, len(buf))
        return buf.value
    except Exception:
        return ""


def _paste_shortcut() -> str:
    # Classic console windows often ignore synthetic Ctrl+V; Shift+Insert is
    # the most compatible paste chord for conhost/cmd/PowerShell prompts.
    if _active_window_class() == "ConsoleWindowClass":
        return "shift+insert"
    return "ctrl+v"


def _release_modifiers(active_modifiers: tuple[str, ...] = ()):
    """Force-release modifier keys that are actually held.

    Releasing keys that are NOT pressed (e.g. ``windows`` when no Win is held)
    can trigger the Start menu on Windows 10/11. We therefore release only
    the modifiers from the active hotkey, or fall back to checking every
    canonical modifier when none is supplied.
    """
    candidates = normalize_all(active_modifiers) if active_modifiers else CANONICAL_MODIFIERS
    for key in candidates:
        try:
            if keyboard.is_pressed(key):
                keyboard.release(key)
        except Exception:
            pass


def _paste_and_restore(text: str):
    """Save old clipboard, paste new text, restore old.

    Moves the slow ``pyperclip.paste()`` Win32 clipboard read off the hot
    keyboard-hook thread. On a contended clipboard this read can stall for
    hundreds of milliseconds (Office, password managers, etc. holding the
    clipboard open). The paste itself still happens in the worker, but it
    runs the instant the worker starts — no measurable user-visible delay.
    """

    with _PASTE_LOCK:
        try:
            old = pyperclip.paste()
        except Exception:
            old = ""

        try:
            # Copy dictated text (trailing space for natural continuation)
            pyperclip.copy(text + " ")
            # keyboard.send has no built-in PAUSE (saves ~200 ms vs pyautogui)
            keyboard.send(_paste_shortcut())
        except Exception as e:
            log.warning("Kunde inte skicka Ctrl+V: %s", e)
            return

        # Grace period so the target app finishes its paste before we
        # overwrite the clipboard.
        time.sleep(_RESTORE_GRACE_SEC)
        for attempt in range(1, _RESTORE_ATTEMPTS + 1):
            try:
                pyperclip.copy(old)
                return
            except Exception as e:
                if attempt == _RESTORE_ATTEMPTS:
                    log.warning("Kunde inte aterstalla urklipp efter paste: %s", e)
                    return
                time.sleep(_RESTORE_DELAY_SEC)


def _paste_and_restore_async(text: str):
    threading.Thread(target=_paste_and_restore, args=(text,), daemon=True).start()

def paste_text(text: str, active_modifiers: tuple[str, ...] = ()):
    """Paste text at the current cursor position.

    Steps:
      1. Release modifier keys that are actually held (only those from the
         dictation hotkey, to avoid triggering Start menu when releasing Win)
      2. Spawn a serialized async worker so queued dictations can't overlap:
         a) Saves current clipboard content (slow on contended systems)
         b) Copies dictated text to clipboard
         c) Sends Ctrl+V
         d) Restores original clipboard
    """
    text = text.strip()
    if not text:
        return

    # Release modifiers synchronously — must happen before Ctrl+V is sent.
    _release_modifiers(active_modifiers)
    _paste_and_restore_async(text)
