"""
freewispr-fast — Svensk speech-to-text för Windows (Parakeet/streaming experimental fork)
Entry point: system tray icon + dictation mode.
"""
import logging
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Logging — basic config now, file handler is wired in main()
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("freewispr")

import config as cfg_module  # noqa: E402  (need APP_NAME before logging dir)

APP_NAME = cfg_module.APP_NAME
APP_DISPLAY_NAME = "freewispr-fast"

log.info("=== %s startar ===", APP_DISPLAY_NAME)

_LOG_DIR = Path.home() / f".{APP_NAME}"
_LOG_FILE = _LOG_DIR / "freewispr.log"


def _attach_file_logging() -> None:
    """Create the log dir and attach the rotating file handler.

    Done from main() (not at import) so unit tests can ``import main`` /
    transitively pull config without writing to disk.
    """
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logging.getLogger().addHandler(handler)
    except Exception as e:
        log.warning("Kunde inte aktivera filloggning: %s", e)

try:
    import threading
    import tkinter as tk

    import pystray
    from PIL import Image, ImageDraw

    # cfg_module is imported above (need APP_NAME for log dir).
    # Heavy modules (torch, faster_whisper, scipy) are imported lazily
    # inside _load_app() so the tray icon appears in <1 second.
    from ui import DictionaryWindow, FloatingIndicator, SettingsWindow, SnippetsWindow, _style
    log.info("Snabb-imports OK")
except Exception:
    log.critical("Import kraschade", exc_info=True)
    sys.exit(1)

# --------------------------------------------------------------------------- #
#  Globals                                                                     #
# --------------------------------------------------------------------------- #

_config: dict = {}
_transcriber = None   # Transcriber (lazy-imported)
_dictation = None     # DictationMode (lazy-imported)
_tray_icon: pystray.Icon | None = None
_tk_root: tk.Tk | None = None
_status_var: tk.StringVar | None = None
_indicator: FloatingIndicator | None = None

# Serializes settings-driven reloads. Without it, a user spamming Save
# could spawn two _reload threads, double-loading the model into VRAM
# and leaking a Transcriber + DictationMode pair.
_reload_lock = threading.Lock()

# Serializes _apply_settings itself so two concurrent Save clicks can't
# interleave config mutations / dictation restarts. The tray menu can
# fire Save from a different thread than tkinter, and _reload_lock alone
# only protects the model-reload branch.
_config_lock = threading.Lock()


# --------------------------------------------------------------------------- #
#  DRY constructors for Transcriber / DictationMode                            #
# --------------------------------------------------------------------------- #

def _make_transcriber(model_size: str, use_cuda: bool):
    """Build a Transcriber from current _config + the given overrides.

    Kept in one place so _load_app, fallback, and reload paths cannot
    drift apart in how they wire up LLM credentials.
    """
    from transcriber import Transcriber
    return Transcriber(
        model_size=model_size,
        use_cuda=use_cuda,
        llm_enabled=(
            _config.get("llm_enabled", False)
            and _config.get("llm_privacy_accepted", False)
        ),
        llm_api_key=_config.get("llm_api_key", ""),
        llm_model=_config.get("llm_model", "gpt-4.1-nano"),
    )


def _make_dictation(transcriber):
    from dictation import DEFAULT_MIN_RMS, DictationMode
    return DictationMode(
        transcriber,
        hotkey=_config.get("hotkey", "ctrl+space"),
        on_status=_set_tray_status,
        indicator=_indicator,
        mic_device=_config.get("mic_device"),
        min_rms=float(_config.get("min_rms", DEFAULT_MIN_RMS)),
        paste_strategy=_config.get("paste_strategy", "auto"),
        paste_threshold=int(_config.get("paste_threshold", 200)),
    )


# --------------------------------------------------------------------------- #
#  Tray icon image — prefer bundled asset, fall back to Pillow-drawn mic      #
# --------------------------------------------------------------------------- #

# When frozen by PyInstaller, assets live next to the executable in _MEIPASS.
_ASSET_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "assets"
_ICON_PATH = _ASSET_DIR / "icon.ico"


def _draw_fallback_icon() -> Image.Image:
    """Mic glyph drawn with Pillow — used only if assets/icon.ico is missing.

    Fork uses orange to be visually distinct from upstream's purple icon.
    """
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size - 4, size - 4], fill="#fc7c5c")  # orange (fork)
    cx = size // 2
    draw.rounded_rectangle([cx - 9, 12, cx + 9, 36], radius=9, fill="white")
    draw.arc([cx - 16, 26, cx + 16, 50], start=0, end=180, fill="white", width=3)
    draw.line([cx, 50, cx, 58], fill="white", width=3)
    draw.line([cx - 8, 58, cx + 8, 58], fill="white", width=3)
    return img


def _make_icon() -> Image.Image:
    """Load the bundled tray icon, fall back to a drawn one on any failure."""
    if _ICON_PATH.is_file():
        try:
            return Image.open(_ICON_PATH)
        except Exception as e:
            log.warning("Kunde inte läsa %s: %s — använder fallback", _ICON_PATH, e)
    return _draw_fallback_icon()


# --------------------------------------------------------------------------- #
#  App init                                                                    #
# --------------------------------------------------------------------------- #

def _load_app():
    global _config, _transcriber, _dictation

    # Lazy-import heavy modules here (runs in background thread)
    # so the tray icon appears instantly.
    log.info("Laddar tunga moduler (torch, whisper, scipy)...")
    log.info("Alla imports OK")

    _config = cfg_module.load()

    model_size = _config.get("model_size", "small")

    # First-run gate: if no model exists for the chosen size, prompt the
    # user to download one before we try to instantiate the transcriber
    # (which would otherwise raise a confusing 'modell saknas' error).
    if not _any_model_present(model_size):
        log.info("Ingen modell hittad — visar FirstRunDialog")
        chosen = _prompt_first_run_modal()
        if chosen is None:
            log.info("Användaren avbröt first-run; avslutar")
            _set_tray_status("Avbruten — avsluta från menyn")
            return
        if chosen != model_size:
            # User picked a different size than what's in config. Persist
            # so the next launch goes straight to _make_transcriber.
            _config["model_size"] = chosen
            try:
                cfg_module.save(_config)
            except Exception as e:
                log.warning("Kunde inte spara model_size efter first-run: %s", e)
            model_size = chosen

    _set_tray_status("Laddar modell...")
    try:
        _transcriber = _make_transcriber(model_size, _config.get("use_cuda", True))
    except Exception as e:
        log.error("Modellfel (%s): %s", model_size, e, exc_info=True)
        log.info("Försöker fallback till 'small' med CPU...")
        _set_tray_status("Modellfel — fallback till 'small'")
        try:
            _transcriber = _make_transcriber("small", False)
            # Update config so we don't crash again next time
            _config["model_size"] = "small"
            _config["use_cuda"] = False
            cfg_module.save(_config)
        except Exception as e2:
            log.error("Även fallback misslyckades: %s", e2, exc_info=True)
            _set_tray_status("FEL: Kunde inte ladda någon modell")
            return
    log.info("Modell laddad! Appen är redo.")

    _dictation = _make_dictation(_transcriber)
    _dictation.start()
    _set_tray_status(f"Klar — håll {_config.get('hotkey','ctrl+space').upper()} för att prata")


def _any_model_present(preferred: str) -> bool:
    """Quick disk check: is at least one Whisper model already on disk?

    We prefer the size in config, but accept any size — the user may have
    converted a different size before and we shouldn't re-prompt.
    """
    try:
        from model_ui import model_is_local
    except Exception:
        # If model_ui can't import (e.g. transcriber import error), don't
        # block startup — let _make_transcriber surface the real error.
        return True
    if model_is_local(preferred):
        return True
    return any(model_is_local(s) for s in ("tiny", "base", "small", "medium", "large"))


def _prompt_first_run_modal() -> str | None:
    """Show FirstRunDialog on the Tk thread, block until user picks/cancels.

    Returns the chosen model size, or None if cancelled.
    The dialog runs its own mainloop, so we drive it via tk_root.after
    and wait on a threading.Event for the result.
    """
    result: dict[str, str | None] = {}
    done = threading.Event()

    def _show():
        try:
            from model_ui import FirstRunDialog
            # Build a transient Toplevel-style flow: use a fresh Toplevel
            # under _tk_root rather than a second Tk root (which would crash
            # pystray's icon thread on shutdown).
            dlg = FirstRunDialog.__new__(FirstRunDialog)
            # Build manually so we attach to existing _tk_root instead of
            # spawning a second mainloop.
            from model_ui import _BaseModelWindow
            _BaseModelWindow.__init__(dlg, parent=_tk_root,
                                      title="freewispr-fast — välj modell",
                                      size=(440, 320))
            dlg._chosen = None
            dlg._size_var = tk.StringVar(value=FirstRunDialog.DEFAULT_PICK)
            dlg._status_var = tk.StringVar(value="")
            dlg._busy = False
            dlg._build()
            dlg.win.transient(_tk_root)
            dlg.win.grab_set()

            def _on_close():
                # Mirror the dialog's own _cancel logic but also signal done.
                if dlg._busy:
                    dlg._status_var.set("Vänta tills nedladdningen är klar innan du avbryter.")
                    return
                dlg._chosen = None
                dlg.win.destroy()

            dlg.win.protocol("WM_DELETE_WINDOW", _on_close)
            dlg.win.bind("<Destroy>", lambda e, d=dlg: (
                result.setdefault("size", d._chosen),
                done.set(),
            ) if e.widget is d.win else None)
        except Exception as e:
            log.error("Kunde inte visa FirstRunDialog: %s", e, exc_info=True)
            result["size"] = None
            done.set()

    _tk_root.after(0, _show)
    done.wait()
    return result.get("size")

# --------------------------------------------------------------------------- #
#  Status helpers                                                              #
# --------------------------------------------------------------------------- #

def _set_tray_status(msg: str):
    if _tray_icon:
        _tray_icon.title = f"{APP_DISPLAY_NAME} — {msg}"
    if _status_var and _tk_root:
        _tk_root.after(0, lambda: _status_var.set(msg))


# --------------------------------------------------------------------------- #
#  Tray menu callbacks                                                         #
# --------------------------------------------------------------------------- #

def _open_snippets(_=None):
    if _tk_root:
        _tk_root.after(0, lambda: SnippetsWindow())


def _open_dictionary(_=None):
    if _tk_root:
        _tk_root.after(0, lambda: DictionaryWindow())


def _open_settings(_=None):
    if _tk_root:
        _tk_root.after(0, _show_settings)


def _show_settings():
    SettingsWindow(_config, on_save=_apply_settings)


def _apply_settings(new_cfg: dict):
    """Validated settings update. Serialised on _config_lock so two
    rapid Save clicks can't interleave mutations / dictation restarts."""
    with _config_lock:
        return _apply_settings_locked(new_cfg)


def _apply_settings_locked(new_cfg: dict):
    global _config, _dictation, _transcriber

    if _reload_lock.locked():
        log.warning("Modellomladdning pågår redan — avvisar nya inställningar")
        _set_tray_status("Vänta: modell laddas fortfarande")
        if _indicator:
            _indicator.show("Vänta: modell laddas", state="error")
            _indicator.hide(delay_ms=2000)
        return False

    old_config = dict(_config)  # shallow copy for rollback
    old_model = _config.get("model_size")
    old_cuda = _config.get("use_cuda")
    old_llm = (_config.get("llm_enabled"), _config.get("llm_api_key"),
               _config.get("llm_model"))

    # Apply in-memory first; persist to disk only after the change is
    # validated (model loaded, dictation rebuilt, etc.). This way a failed
    # reload doesn't leave a broken config on disk for the next launch.
    _config.update(new_cfg)

    def _persist() -> bool:
        try:
            cfg_module.save(_config)
            return True
        except Exception as e:
            log.error("Kunde inte spara installningar: %s", e, exc_info=True)
            _set_tray_status("Fel: kunde inte spara installningar")
            if _indicator:
                _indicator.show("Kunde inte spara installningar", state="error")
                _indicator.hide(delay_ms=4000)
            return False

    def _rollback():
        _config.clear()
        _config.update(old_config)

    new_model = _config.get("model_size", "small")
    new_cuda = _config.get("use_cuda", True)
    new_llm = (_config.get("llm_enabled"), _config.get("llm_api_key"),
               _config.get("llm_model"))

    model_changed = (old_model != new_model) or (old_cuda != new_cuda)
    llm_changed = old_llm != new_llm

    # Fast path: LLM-only change. Mutate the existing transcriber in place
    # so we don't pay 5-15 s + extra VRAM for a full model reload.
    if llm_changed and not model_changed and _transcriber is not None:
        old_transcriber_llm = (
            _transcriber.llm_enabled,
            _transcriber.llm_api_key,
            _transcriber.llm_model,
        )
        _transcriber.llm_enabled = (
            _config.get("llm_enabled", False)
            and _config.get("llm_privacy_accepted", False)
        )
        _transcriber.llm_api_key = _config.get("llm_api_key", "")
        _transcriber.llm_model = _config.get("llm_model", "gpt-4.1-nano")
        log.info("LLM-inställningar uppdaterade i befintlig transcriber")
        # Hotkey/mic may still have changed; rebuild dictation cheaply.
        _restart_dictation()
        if not _persist():
            _rollback()
            (
                _transcriber.llm_enabled,
                _transcriber.llm_api_key,
                _transcriber.llm_model,
            ) = old_transcriber_llm
            _restart_dictation()
            return False
        _set_tray_status(
            f"Inställningar sparade — håll {_config.get('hotkey','ctrl+space').upper()}"
        )
        return True

    if model_changed:
        # Acquire before returning to the event loop so a second Save cannot
        # mutate _config in the gap before the background thread starts.
        if not _reload_lock.acquire(blocking=False):
            log.warning("Modellomladdning pågår redan — avvisar denna")
            _rollback()
            _set_tray_status("Vänta: modell laddas fortfarande")
            if _indicator:
                _indicator.show("Vänta: modell laddas", state="error")
                _indicator.hide(delay_ms=2000)
            return False
        _set_tray_status(f"Laddar modell '{new_model}'...")
        if _indicator:
            _indicator.show(f"Laddar modell '{new_model}'...", state="transcribe")

        def _reload():
            global _transcriber, _dictation
            try:
                old_transcriber = _transcriber
                old_dictation = _dictation
                try:
                    new_transcriber = _make_transcriber(new_model, new_cuda)
                except Exception as e:
                    log.error("Fel vid modellbyte: %s", e, exc_info=True)
                    _set_tray_status("Modellfel — använder tidigare modell")
                    if _indicator:
                        _indicator.show(f"Modellfel: {e}", state="error")
                        _indicator.hide(delay_ms=4000)
                    # Roll back the in-memory config so the next Save attempt
                    # sees the real previous state.
                    _rollback()
                    return
                # Unhook old hotkeys before starting the new DictationMode so
                # two hook sets cannot record/paste concurrently. Do not wait
                # for the old worker here; close() below will wait on the
                # model lock if a transcription is still in flight.
                if old_dictation is not None:
                    try:
                        old_dictation.stop(wait=False)
                    except Exception as e:
                        log.debug("Kunde inte stoppa gammal dictation rent: %s", e)
                _transcriber = new_transcriber
                _dictation = _make_dictation(_transcriber)
                _dictation.start()
                # Cleanup old model off the reload path. close() may wait for
                # an in-flight transcription; don't block settings/UI or keep
                # _reload_lock held for that duration.
                def _cleanup_old():
                    if old_transcriber is not None:
                        try:
                            old_transcriber.close()
                        except Exception as e:
                            log.debug("Kunde inte stänga gammal transcriber: %s", e)

                threading.Thread(target=_cleanup_old, daemon=True).start()
                log.info("Modell '%s' laddad!", new_model)
                # Model loaded OK — now it's safe to persist the new config.
                if not _persist():
                    # Rare: disk write failed after a successful reload.
                    # Leave the running app on the new model (already loaded)
                    # but warn the user that the next launch will revert.
                    log.warning("Modell laddad men config kunde inte sparas")
                _set_tray_status(
                    f"Modell '{new_model}' klar — håll {_config.get('hotkey','ctrl+space').upper()}"
                )
                if _indicator:
                    _indicator.show(f"Modell '{new_model}' klar", state="done")
                    _indicator.hide(delay_ms=2000)
            finally:
                _reload_lock.release()

        threading.Thread(target=_reload, daemon=True).start()
        return True

    # No model/LLM change — just hotkey/mic. Restart dictation cheaply.
    _restart_dictation()
    if not _persist():
        _rollback()
        _restart_dictation()  # rebind to rolled-back hotkey/mic
        return False
    _set_tray_status(
        f"Inställningar sparade — håll {_config.get('hotkey','ctrl+space').upper()} för att prata"
    )
    return True


def _restart_dictation():
    """Stop the current DictationMode (if any) and start a fresh one
    bound to the current _transcriber + _config."""
    global _dictation
    if _dictation:
        _dictation.stop()
    if _transcriber is None:
        return
    _dictation = _make_dictation(_transcriber)
    _dictation.start()


def _startup_exe_path() -> str:
    """Return the command to register for startup."""
    import os
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller exe — register the exe directly
        return f'"{sys.executable}"'
    else:
        # Running as script — use pythonw to avoid console window
        script = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if not os.path.exists(pythonw):
            pythonw = sys.executable
        return f'"{pythonw}" "{script}"'


_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE = APP_NAME  # "freewispr-swedish-parakeet" — distinct from upstream


def _open_run_key(write: bool = False):
    """Return an open HKCU\\...\\Run registry key. Caller must CloseKey."""
    import winreg
    access = winreg.KEY_SET_VALUE if write else winreg.KEY_READ
    return winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, access)


def _is_startup_enabled() -> bool:
    import winreg
    try:
        key = _open_run_key(write=False)
        try:
            winreg.QueryValueEx(key, _RUN_VALUE)
        finally:
            winreg.CloseKey(key)
        return True
    except Exception:
        return False


def _enable_startup():
    import winreg
    key = _open_run_key(write=True)
    try:
        winreg.SetValueEx(key, _RUN_VALUE, 0, winreg.REG_SZ, _startup_exe_path())
    finally:
        winreg.CloseKey(key)


def _toggle_startup(_=None):
    import winreg
    try:
        key = _open_run_key(write=True)
        try:
            if _is_startup_enabled():
                winreg.DeleteValue(key, _RUN_VALUE)
                _set_tray_status("Borttagen från uppstart")
            else:
                winreg.SetValueEx(key, _RUN_VALUE, 0, winreg.REG_SZ, _startup_exe_path())
                _set_tray_status("Startar med Windows ✓")
        finally:
            winreg.CloseKey(key)
    except OSError as e:
        # Locked-down corporate machines, anti-virus quarantine, or
        # group-policy restrictions can deny HKCU\\...\\Run access.
        # Surface the failure in the tray instead of crashing the app.
        log.error("Kunde inte uppdatera autostart: %s", e)
        _set_tray_status("Fel: kunde inte uppdatera autostart")
        if _indicator:
            _indicator.show("Kunde inte uppdatera autostart", state="error")
            _indicator.hide(delay_ms=4000)
        return
    _rebuild_menu()


def _rebuild_menu():
    if _tray_icon:
        _tray_icon.menu = _build_menu()


def _build_menu():
    startup_label = "✓ Starta med Windows" if _is_startup_enabled() else "Starta med Windows"
    return pystray.Menu(
        pystray.MenuItem("Snippets", _open_snippets),
        pystray.MenuItem("Personlig ordlista", _open_dictionary),
        pystray.MenuItem("Inställningar", _open_settings),
        pystray.MenuItem("Hantera modeller", _open_model_manager),
        pystray.MenuItem("Sekretess och data", _open_privacy),
        pystray.MenuItem(startup_label, _toggle_startup),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(f"Avsluta {APP_DISPLAY_NAME}", _quit),
    )


def _open_model_manager(_=None):
    if _tk_root:
        active = _config.get("model_size", "") if _config else ""
        _tk_root.after(0, lambda: _show_model_manager(active))


def _show_model_manager(active: str):
    from model_ui import ModelManagerWindow
    ModelManagerWindow(_tk_root, active_model=active)


def _open_privacy(_=None):
    if _tk_root:
        _tk_root.after(0, _show_privacy)


def _show_privacy():
    from privacy_ui import PrivacyWindow
    PrivacyWindow(_tk_root)


def _quit(_=None):
    if _dictation:
        _dictation.stop()
    if _tray_icon:
        _tray_icon.stop()
    if _tk_root:
        _tk_root.quit()
        _tk_root.destroy()
    sys.exit(0)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    global _tray_icon, _tk_root, _status_var, _indicator

    # Wire up file logging now (deferred from import time so tests can import
    # this module without touching ~/.freewispr-swedish/).
    _attach_file_logging()

    # Hidden tk root — keeps tkinter event loop running for Toplevel windows
    _tk_root = tk.Tk()
    _tk_root.withdraw()
    _style(_tk_root)

    _status_var = tk.StringVar(value="Startar...")
    _indicator = FloatingIndicator(_tk_root)

    # Build tray icon
    menu = _build_menu()
    _tray_icon = pystray.Icon(
        APP_NAME,
        _make_icon(),
        f"{APP_DISPLAY_NAME} — Startar...",
        menu,
    )

    # Load model in background so the tray appears immediately
    threading.Thread(target=_load_app, daemon=True).start()

    # Run tray in a background thread; tkinter runs on main thread
    tray_thread = threading.Thread(target=_tray_icon.run, daemon=True)
    tray_thread.start()

    # tkinter main loop (needed for Toplevel windows + FloatingIndicator)
    _tk_root.mainloop()


if __name__ == "__main__":
    main()
