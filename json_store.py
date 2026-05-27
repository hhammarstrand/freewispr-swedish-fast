import json
import logging
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import strftime
from typing import TypeVar

log = logging.getLogger("freewispr")

_D = TypeVar("_D", bound=dict)


class JsonCache:
    """Mtime-based cache for a single JSON file containing a dict.

    Provides load() and save() with automatic in-memory caching:
    the file is only re-read when its mtime changes. save() updates
    the cache immediately so a subsequent load() is always consistent.

    Usage::

        _store = JsonCache(Path.home() / ".app" / "data.json")
        data = _store.load()        # dict
        _store.save({...})
        ts = _store.mtime()         # float, 0.0 if missing
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict | None = None
        self._cache_mtime: float = 0.0

    def mtime(self) -> float:
        try:
            return self._path.stat().st_mtime
        except OSError:
            return 0.0

    def load(self) -> dict:
        current_mt = self.mtime()
        if self._cache is not None and current_mt == self._cache_mtime:
            return self._cache
        if self._path.exists():
            self._cache = dict(load_json(self._path, {}))
            self._cache_mtime = self.mtime()
        else:
            self._cache = {}
            self._cache_mtime = current_mt
        return self._cache

    def save(self, data: dict) -> None:
        save_json_atomic(self._path, data)
        self._cache = data
        self._cache_mtime = self.mtime()


def load_json(path: Path, default):
    if not path.exists():
        return default.copy() if isinstance(default, dict) else default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if path.name == "config.json":
            log.warning("Korrupt JSON i %s; hoppar over backup for config", path)
            return default.copy() if isinstance(default, dict) else default
        backup = path.with_suffix(path.suffix + f".corrupt-{strftime('%Y%m%d-%H%M%S')}")
        try:
            shutil.copy2(path, backup)
            log.warning("Korrupt JSON i %s; backup sparad som %s", path, backup)
        except Exception as backup_error:
            log.warning("Korrupt JSON i %s; kunde inte skapa backup: %s", path, backup_error)
        log.warning("Anvander standarddata efter JSON-fel i %s: %s", path, e)
        return default.copy() if isinstance(default, dict) else default
    except Exception as e:
        log.warning("Kunde inte lasa JSON fran %s: %s", path, e)
        return default.copy() if isinstance(default, dict) else default


def save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_name = f.name
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
