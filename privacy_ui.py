"""Privacy / data-cleanup UI for freewispr-fast.

PR 1.4. Lets the user inspect and selectively wipe local state:

- freewispr.log         (transcription metadata + errors)
- learned.json          (auto-learned corrections from snippet rewrites)
- corrections.json      (manual word-correction dictionary)
- snippets.json         (text expansion triggers)
- hotwords.txt          (optional extra hotwords for the decoder)
- models cache          (downloaded Whisper weights — large)

We deliberately do NOT touch config.json (it stores hotkey, model size
etc; nuking it would not be a privacy concern but would surprise the
user) or the Windows Credential Manager entry for the LLM API key
(handled separately by the LLM settings UI).

There is also a "Show data folder" button that opens explorer.exe on the
config dir so the user can verify what's there.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from config import CONFIG_DIR

log = logging.getLogger("privacy_ui")


BG = "#0f0f0f"
ACC = "#7c5cfc"
FG = "#e8e8e8"
FG2 = "#888"
FONT = ("Segoe UI", 10)


# Map of UI label -> (path-on-disk, short description).
# Paths are computed lazily so tests can monkeypatch CONFIG_DIR.
def _data_targets() -> dict[str, tuple[Path, str]]:
    return {
        "Loggfil":          (CONFIG_DIR / "freewispr.log",
                             "Felmeddelanden och transkriberings-metadata."),
        "Lärda rättelser":  (CONFIG_DIR / "learned.json",
                             "Auto-lärda omskrivningar (PR 2.5)."),
        "Manuell ordlista": (CONFIG_DIR / "corrections.json",
                             "Dina manuellt tillagda rättelser."),
        "Snippets":         (CONFIG_DIR / "snippets.json",
                             "Dina text-expansioner."),
        "Hotwords":         (CONFIG_DIR / "hotwords.txt",
                             "Extra ord till decodern (om filen finns)."),
        "Modellcache":      (CONFIG_DIR / "models",
                             "Nedladdade Whisper-modeller (kan vara flera GB)."),
    }


def _path_size_human(path: Path) -> str:
    """Return '12.3 MB' or '— saknas' for a file or directory."""
    if not path.exists():
        return "— saknas"
    try:
        if path.is_file():
            n = path.stat().st_size
        else:
            n = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError as e:
        log.debug("Kunde inte mäta %s: %s", path, e)
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def wipe_path(path: Path) -> bool:
    """Delete a file or directory. Returns True if anything was removed.

    Raises OSError if the target exists but cannot be removed (file lock,
    permissions). Caller surfaces the error to the user.
    """
    if not path.exists():
        return False
    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    log.info("Rensade %s", path)
    return True


def open_data_folder() -> None:
    """Open Explorer/Finder on the config dir. No-ops on unknown platforms."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(str(CONFIG_DIR))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(CONFIG_DIR)], check=False)
        else:
            subprocess.run(["xdg-open", str(CONFIG_DIR)], check=False)
    except OSError as e:
        log.warning("Kunde inte öppna datamapp: %s", e)


# --------------------------------------------------------------------------- #
#  Window                                                                      #
# --------------------------------------------------------------------------- #


class PrivacyWindow:
    """Toplevel window with one row per data target plus a 'show folder' button."""

    def __init__(self, parent: tk.Misc | None = None):
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.title("Sekretess och data")
        self.win.configure(bg=BG)
        self.win.geometry("560x460")
        self.win.resizable(False, False)

        style = ttk.Style(self.win)
        try:
            style.theme_use("clam")
        except tk.TclError:
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

        self._rows: dict[str, dict] = {}
        self._status_var = tk.StringVar(value="")
        self._build()

    def _build(self) -> None:
        ttk.Label(self.win, text="Sekretess och data",
                  font=("Segoe UI", 13, "bold")).pack(pady=(14, 2))
        ttk.Label(self.win,
                  text=f"All data lagras lokalt i {CONFIG_DIR}",
                  style="Sub.TLabel", wraplength=520,
                  justify="left").pack(padx=20, pady=(0, 12))

        table = ttk.Frame(self.win)
        table.pack(padx=20, fill="x")

        for label, (path, desc) in _data_targets().items():
            row = ttk.Frame(table)
            row.pack(fill="x", pady=4)

            ttk.Label(row, text=label, width=18).pack(side="left", anchor="n")
            text_col = ttk.Frame(row)
            text_col.pack(side="left", fill="x", expand=True)
            size_lbl = ttk.Label(text_col, text=_path_size_human(path),
                                 style="Sub.TLabel")
            size_lbl.pack(anchor="w")
            ttk.Label(text_col, text=desc, style="Sub.TLabel",
                      wraplength=300, justify="left").pack(anchor="w")

            btn = ttk.Button(
                row, text="Rensa", style="Danger.TButton",
                command=lambda p=path, l=label: self._on_wipe(l, p),
            )
            btn.pack(side="right", anchor="n")

            self._rows[label] = {"size": size_lbl, "btn": btn, "path": path}

        ttk.Label(self.win, textvariable=self._status_var,
                  style="Sub.TLabel", wraplength=520,
                  justify="left").pack(padx=20, pady=(10, 0), fill="x")

        bottom = ttk.Frame(self.win)
        bottom.pack(side="bottom", fill="x", padx=20, pady=12)
        ttk.Button(bottom, text="Visa datamapp",
                   command=open_data_folder).pack(side="left")
        ttk.Button(bottom, text="Stäng",
                   command=self.win.destroy).pack(side="right")

    def _refresh_row(self, label: str) -> None:
        row = self._rows[label]
        row["size"].configure(text=_path_size_human(row["path"]))

    def _on_wipe(self, label: str, path: Path) -> None:
        if not messagebox.askyesno(
            "Bekräfta",
            f"Ta bort {label}?\n\n{path}",
            parent=self.win,
        ):
            return
        try:
            removed = wipe_path(path)
        except OSError as e:
            self._status_var.set(
                f"Kunde inte ta bort {label}: {e}. Är filen i bruk?"
            )
            return
        if removed:
            self._refresh_row(label)
            self._status_var.set(f"{label} rensad.")
        else:
            self._status_var.set(f"{label} fanns inte (inget att rensa).")
