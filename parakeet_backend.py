"""Parakeet TDT backend wrapping NVIDIA NeMo parakeet-tdt-0.6b-v3.

NeMo is an optional dependency — this module imports cleanly even when NeMo
is not installed. Call is_available() before constructing ParakeetBackend.
"""

import gc
import logging
import os
import tempfile
import threading
import time

import numpy as np

log = logging.getLogger("freewispr")

_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def is_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import nemo.collections.asr  # noqa: F401
        return True
    except Exception:
        return False


class ParakeetBackend:
    def __init__(self, device: str = "cuda") -> None:
        self._model_lock = threading.RLock()
        self.device = device

        t0 = time.monotonic()
        from nemo.collections.asr import models as nemo_asr
        self.model = nemo_asr.ASRModel.from_pretrained(_MODEL_ID)
        self.model = self.model.to(device)
        self.model.eval()
        log.info("Parakeet '%s' laddad OK [%s] pa %.0f ms",
                 _MODEL_ID, device, (time.monotonic() - t0) * 1000)

        self._warmed = False
        threading.Thread(target=self.warmup, name="parakeet-warmup",
                         daemon=True).start()

    def transcribe_raw(self, audio: np.ndarray) -> str:
        from scipy.io import wavfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            wavfile.write(tmp_path, 16000, audio.astype(np.float32))

            with self._model_lock:
                results = self.model.transcribe([tmp_path])

            if not results:
                return ""
            r = results[0]
            # NeMo 2.x returns list[Hypothesis] with a .text attribute;
            # older releases return list[str].
            text = r.text if hasattr(r, "text") else r
            return text.strip()
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def warmup(self) -> None:
        try:
            t0 = time.monotonic()
            silent = np.zeros(16000, dtype=np.float32)
            with self._model_lock:
                model = self.model
                if model is None:
                    return
            self.transcribe_raw(silent)
            self._warmed = True
            log.info("Parakeet-warmup klar pa %.0f ms", (time.monotonic() - t0) * 1000)
        except Exception as e:
            log.debug("Parakeet-warmup misslyckades (ignoreras): %s", e)

    def close(self) -> None:
        try:
            with self._model_lock:
                model = getattr(self, "model", None)
                if model is None:
                    return
                self.model = None  # type: ignore[assignment]
                del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            log.debug("Kunde inte frigora Parakeet-modell rent: %s", e)
