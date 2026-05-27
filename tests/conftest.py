"""Test-time stubs for OS-level deps that aren't available in CI.

The test suite monkeypatches things like `keyboard.send` and
`pyperclip.copy` already, but several modules import OS-bound libraries
at module import time (sounddevice, pystray's display backends, etc).
On Linux CI runners these either aren't installed or can't initialize
(no X display), which causes a cascade of unrelated test failures.

We stub these modules in sys.modules BEFORE pytest starts collecting,
so any `import sounddevice` from production code transparently picks
up the stub. Locally on Windows the real packages are already imported
by the time conftest runs and we don't touch them.

Scope: only stub things that are NOT covered by existing per-test
monkeypatches in test_core_logic.py. Don't stub keyboard / pyperclip /
torch — those tests already monkeypatch them and a global stub would
mask real test failures.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace


def _stub(name: str, **attrs) -> None:
    """Insert a placeholder module if and only if the real one is absent.

    Local dev (Windows) always has the real packages and we want to
    exercise real code paths there. CI (Ubuntu) doesn't, so we substitute
    a minimal namespace.
    """
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = SimpleNamespace(**attrs)


# --------------------------------------------------------------------------- #
#  Audio                                                                       #
# --------------------------------------------------------------------------- #

# sounddevice is imported at module level by audio.py and dictation.py.
# We give it the minimum surface that import-time code paths touch.
def _fake_input_stream(*args, **kwargs):
    """Return an object with start/stop/close so 'with InputStream(...)' works."""
    return SimpleNamespace(
        start=lambda: None,
        stop=lambda: None,
        close=lambda: None,
        __enter__=lambda self=None: self,
        __exit__=lambda *a, **kw: None,
    )


_stub(
    "sounddevice",
    InputStream=_fake_input_stream,
    query_devices=lambda *a, **kw: [],
    default=SimpleNamespace(device=(None, None), samplerate=16000),
    PortAudioError=Exception,
)


# --------------------------------------------------------------------------- #
#  Tray / system UI                                                            #
# --------------------------------------------------------------------------- #

# pystray pulls in Xlib on Linux which dies without DISPLAY. Tests that
# need real pystray already 'pytest.importorskip("pystray")', so a stub
# here is safe: those tests get skipped, the rest run.
class _FakeIcon:
    def __init__(self, *args, **kwargs):
        self.title = ""
        self.menu = None

    def run(self):
        pass

    def stop(self):
        pass


_stub(
    "pystray",
    Icon=_FakeIcon,
    Menu=SimpleNamespace(SEPARATOR=object()),
    MenuItem=lambda *a, **kw: SimpleNamespace(*a, **kw)
    if False
    else SimpleNamespace(text=a[0] if a else "", action=a[1] if len(a) > 1 else None),
)


# --------------------------------------------------------------------------- #
#  Credential storage                                                          #
# --------------------------------------------------------------------------- #

_stub(
    "keyring",
    get_password=lambda service, user: None,
    set_password=lambda service, user, pw: None,
    delete_password=lambda service, user: None,
)


# --------------------------------------------------------------------------- #
#  Tkinter — on Ubuntu CI without xvfb, Tk() raises TclError.                  #
# --------------------------------------------------------------------------- #

# We do NOT stub tkinter wholesale (too invasive — many tests just import
# the module). Instead we mark the two settings tests that actually call
# Tk() to skip on systems without a display. They already exist; we hook
# into pytest_collection_modifyitems below.


import os  # noqa: E402

import pytest  # noqa: E402


_TESTS_REQUIRING_DISPLAY = {
    "test_main_apply_settings_serialised",
    "test_llm_only_save_failure_restores_transcriber_state",
}


def pytest_collection_modifyitems(config, items):
    """Skip Tk-using tests on headless Linux (no DISPLAY)."""
    if sys.platform != "linux":
        return
    if os.environ.get("DISPLAY"):
        return
    skip = pytest.mark.skip(reason="requires X display (headless Linux CI)")
    for item in items:
        if item.name in _TESTS_REQUIRING_DISPLAY:
            item.add_marker(skip)
