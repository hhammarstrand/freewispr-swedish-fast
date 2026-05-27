"""Model download / management UI for freewispr-fast.

Two windows live here, isolated from ui.py to keep that file's surface
area stable:

- FirstRunDialog       — modal shown on first launch when no model exists.
                         User picks a size, we run convert_model.convert()
                         in a worker thread, then close and let main.py
                         continue with _load_app().

- ModelManagerWindow   — opened from the tray menu. Lists known KBLab
                         sizes, marks which are downloaded, lets the user
                         download more or delete unused ones.

Both windows are intentionally small. They depend only on Tkinter,
convert_model, and the MODEL_DIR layout from transcriber.py.
"""
from __future__ import annotations

import logging
import shutil
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import convert_model
from transcriber import MODEL_DIR, _find_local_model

log = logging.getLogger("model_ui")


# Match the palette from ui.py so the two windows feel like the same app.
BG = "#0f0f0f"
BG2 = "#1a1a1a"
ACC = "#7c5cfc"
FG = "#e8e8e8"
FG2 = "#888"
FONT = ("Segoe UI", 10)


# Approximate on-disk sizes in MB after ct2 float16 conversion. Shown to
# the user so they don't pick "large" on a 50 GB SSD without warning.
MODEL_SIZE_MB = {
    "tiny": 80,
    "base": 150,
    "small": 500,
    "medium": 1500,
    "large": 3000,
}


def model_is_local(size: str) -> bool:
    repo = convert_model.KBLAB_MODELS.get(size)
    if not repo:
        return False
    return _find_local_model(repo) is not None


def delete_local_model(size: str) -> bool:
    """Remove both the converted ct2 dir and the HF snapshot cache.

    Returns True if at least one of them existed and was removed.
    """
    short_name = f"kb-whisper-{size}"
    ct2_dir = MODEL_DIR / f"{short_name}-ct2"
    hf_dir = MODEL_DIR / f"models--KBLab--{short_name}"

    removed = False
    for path in (ct2_dir, hf_dir):
        if path.exists():
            try:
                shutil.rmtree(path)
                log.info("Tog bort %s", path)
                removed = True
            except OSError as e:
                # File locked by another process is the common failure here
                # (e.g. the model is loaded right now). Surface a clear error.
                log.error("Kunde inte ta bort %s: %s", path, e)
                raise
    return removed


# --------------------------------------------------------------------------- #
#  Shared base class                                                           #
# --------------------------------------------------------------------------- #


class _BaseModelWindow:
    """Common Tk setup: dark theme, centered, basic ttk styles."""

    def __init__(self, parent: tk.Misc | None, title: str, size: tuple[int, int]):
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title(title)
        self.win.configure(bg=BG)
        w, h = size
        self.win.geometry(f"{w}x{h}")
        self.win.resizable(False, False)

        style = ttk.Style(self.win)
        try:
            style.theme_use("clam")
        except tk.TclError:
            # 'clam' is bundled with cpython; this branch is defensive.
            pass
        style.configure("TButton", background=ACC, foreground=FG, font=FONT,
                        relief="flat", padding=6)
        style.map("TButton", background=[("active", "#5a3fd4")])
        style.configure("Danger.TButton", background="#c0392b", foreground=FG,
                        font=FONT, relief="flat", padding=6)
        style.map("Danger.TButton", background=[("active", "#96281b")])
        style.configure("TLabel", background=BG, foreground=FG, font=FONT)
        style.configure("Sub.TLabel", background=BG, foreground=FG2,
                        font=("Segoe UI", 9))
        style.configure("TFrame", background=BG)


# --------------------------------------------------------------------------- #
#  First-run dialog                                                            #
# --------------------------------------------------------------------------- #


class FirstRunDialog(_BaseModelWindow):
    """Blocks _load_app() until the user picks and downloads a model.

    Usage from main.py before _load_app():

        if not any_model_present():
            size = FirstRunDialog().run_modal()  # returns 'small' etc, or None on cancel
            if size is None:
                sys.exit(0)
            # transcriber.py will now find the freshly downloaded model

    The dialog runs its own Tk mainloop because at first-run time the
    main app's tk root hasn't been built yet.
    """

    DEFAULT_PICK = "small"

    def __init__(self):
        super().__init__(parent=None, title="freewispr-fast — välj modell",
                         size=(440, 320))
        self._chosen: str | None = None
        self._size_var = tk.StringVar(value=self.DEFAULT_PICK)
        self._status_var = tk.StringVar(value="")
        self._busy = False
        self._build()

    def _build(self) -> None:
        ttk.Label(self.win, text="Välkommen till freewispr-fast",
                  font=("Segoe UI", 14, "bold")).pack(pady=(20, 4))
        ttk.Label(self.win,
                  text="Välj en svensk Whisper-modell att ladda ner.",
                  style="Sub.TLabel").pack()
        ttk.Label(self.win,
                  text="Större modell = bättre kvalitet men mer minne och längre laddning.",
                  style="Sub.TLabel").pack(pady=(0, 16))

        radio_frame = ttk.Frame(self.win)
        radio_frame.pack(padx=24, fill="x")
        for size in ("tiny", "base", "small", "medium", "large"):
            mb = MODEL_SIZE_MB[size]
            label = f"{size}  ({mb} MB)"
            if model_is_local(size):
                label += "  ✓ redan nedladdad"
            ttk.Radiobutton(
                radio_frame, text=label, value=size,
                variable=self._size_var,
            ).pack(anchor="w", pady=2)

        self._status_label = ttk.Label(self.win, textvariable=self._status_var,
                                       style="Sub.TLabel", wraplength=400,
                                       justify="left")
        self._status_label.pack(padx=24, pady=(12, 0), fill="x")

        btn_row = ttk.Frame(self.win)
        btn_row.pack(side="bottom", fill="x", padx=24, pady=16)
        ttk.Button(btn_row, text="Avbryt", command=self._cancel,
                   style="Danger.TButton").pack(side="left")
        self._dl_btn = ttk.Button(btn_row, text="Ladda ner och fortsätt",
                                  command=self._on_download)
        self._dl_btn.pack(side="right")

        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

    def _on_download(self) -> None:
        if self._busy:
            return
        size = self._size_var.get()
        if model_is_local(size):
            # Already there — short-circuit, no need to redownload.
            self._chosen = size
            self.win.destroy()
            return
        self._busy = True
        self._dl_btn.configure(state="disabled")
        self._status_var.set(
            f"Laddar ner {size} (~{MODEL_SIZE_MB[size]} MB) — detta kan ta några minuter…"
        )

        def worker() -> None:
            try:
                convert_model.convert(size)
                if not model_is_local(size):
                    # convert() logs and returns silently on missing deps;
                    # treat 'still not present' as failure for the user.
                    self.win.after(0, self._on_failed,
                                   "Nedladdning verkar ha misslyckats. "
                                   "Kontrollera loggen och att 'transformers' "
                                   "och 'ctranslate2' är installerade.")
                    return
                self.win.after(0, self._on_done, size)
            except Exception as e:  # pylint: disable=broad-except
                log.error("Modellnedladdning misslyckades: %s", e, exc_info=True)
                self.win.after(0, self._on_failed, str(e))

        threading.Thread(target=worker, daemon=True,
                         name="model-download").start()

    def _on_done(self, size: str) -> None:
        self._chosen = size
        self.win.destroy()

    def _on_failed(self, msg: str) -> None:
        self._busy = False
        self._dl_btn.configure(state="normal")
        self._status_var.set(f"Fel: {msg}")

    def _cancel(self) -> None:
        if self._busy:
            # Don't let the user yank the rug out from under the worker —
            # convert_model has no cancel hook and a partial download in
            # the cache will confuse the next launch.
            self._status_var.set("Vänta tills nedladdningen är klar innan du avbryter.")
            return
        self._chosen = None
        self.win.destroy()

    def run_modal(self) -> str | None:
        """Block until the user closes the dialog; return chosen size or None."""
        self.win.mainloop()
        return self._chosen


# --------------------------------------------------------------------------- #
#  Manage-models window (opened from tray)                                     #
# --------------------------------------------------------------------------- #


class ModelManagerWindow(_BaseModelWindow):
    """Tray-opened window. Download / delete models; does NOT switch the
    active model — the user does that from the Settings window (which
    triggers the existing reload flow in main._apply_settings)."""

    def __init__(self, parent: tk.Misc, active_model: str = ""):
        super().__init__(parent, "Hantera modeller", (520, 380))
        self._active = active_model
        self._busy = False
        self._row_widgets: dict[str, dict] = {}
        self._status_var = tk.StringVar(value="")
        self._build()

    def _build(self) -> None:
        ttk.Label(self.win, text="Hantera nedladdade modeller",
                  font=("Segoe UI", 13, "bold")).pack(pady=(16, 4))
        ttk.Label(self.win,
                  text="Aktiv modell kan inte tas bort. Byt modell via Inställningar först.",
                  style="Sub.TLabel").pack(pady=(0, 12))

        table = ttk.Frame(self.win)
        table.pack(padx=20, fill="x")

        for size in ("tiny", "base", "small", "medium", "large"):
            row = ttk.Frame(table)
            row.pack(fill="x", pady=3)

            present = model_is_local(size)
            mb = MODEL_SIZE_MB[size]
            status = "✓ nedladdad" if present else "— saknas"
            if size == self._active:
                status += "  (AKTIV)"

            ttk.Label(row, text=f"{size}  ({mb} MB)", width=18).pack(side="left")
            status_lbl = ttk.Label(row, text=status, style="Sub.TLabel", width=24)
            status_lbl.pack(side="left")

            if present:
                btn = ttk.Button(
                    row, text="Ta bort", style="Danger.TButton",
                    command=lambda s=size: self._on_delete(s),
                )
                if size == self._active:
                    btn.configure(state="disabled")
            else:
                btn = ttk.Button(
                    row, text="Ladda ner",
                    command=lambda s=size: self._on_download(s),
                )
            btn.pack(side="right")

            self._row_widgets[size] = {"status": status_lbl, "btn": btn}

        ttk.Label(self.win, textvariable=self._status_var,
                  style="Sub.TLabel", wraplength=480,
                  justify="left").pack(padx=20, pady=(12, 0), fill="x")

        ttk.Button(self.win, text="Stäng", command=self.win.destroy).pack(
            side="bottom", pady=12
        )

    def _refresh_row(self, size: str) -> None:
        present = model_is_local(size)
        status = "✓ nedladdad" if present else "— saknas"
        if size == self._active:
            status += "  (AKTIV)"
        w = self._row_widgets[size]
        w["status"].configure(text=status)
        btn = w["btn"]
        # Rebuild the button text/command in place to avoid relayout.
        if present:
            btn.configure(text="Ta bort", style="Danger.TButton",
                          command=lambda s=size: self._on_delete(s))
            if size == self._active:
                btn.configure(state="disabled")
            else:
                btn.configure(state="normal")
        else:
            btn.configure(text="Ladda ner", style="TButton",
                          command=lambda s=size: self._on_download(s),
                          state="normal")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for size, widgets in self._row_widgets.items():
            # Active model's delete stays disabled regardless.
            if not busy and size == self._active and widgets["btn"].cget("text") == "Ta bort":
                widgets["btn"].configure(state="disabled")
            else:
                widgets["btn"].configure(state=state)

    def _on_download(self, size: str) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self._status_var.set(f"Laddar ner {size}…")

        def worker() -> None:
            try:
                convert_model.convert(size)
                self.win.after(0, self._after_download, size, None)
            except Exception as e:  # pylint: disable=broad-except
                log.error("Nedladdning misslyckades: %s", e, exc_info=True)
                self.win.after(0, self._after_download, size, str(e))

        threading.Thread(target=worker, daemon=True,
                         name=f"model-dl-{size}").start()

    def _after_download(self, size: str, err: str | None) -> None:
        self._set_busy(False)
        if err:
            self._status_var.set(f"Fel: {err}")
            return
        self._refresh_row(size)
        self._status_var.set(f"{size} klar.")

    def _on_delete(self, size: str) -> None:
        if self._busy:
            return
        if size == self._active:
            # Should not happen — button is disabled — but defend anyway.
            return
        if not messagebox.askyesno(
            "Bekräfta",
            f"Ta bort {size}-modellen från disk?",
            parent=self.win,
        ):
            return
        try:
            delete_local_model(size)
        except OSError as e:
            self._status_var.set(
                f"Kunde inte ta bort {size}: {e}. Är modellen kanske igång?"
            )
            return
        self._refresh_row(size)
        self._status_var.set(f"{size} borttagen.")
