"""STT — Whisper large-v3-turbo via pywhispercpp (whisper.cpp).

Contratto (spec §6.3): ``transcribe(audio) -> str``. Su Apple Silicon, se il
pacchetto è compilato con ``WHISPER_COREML=1``, l'encoder gira sul Neural Engine
(ANE), lasciando la GPU libera per l'LLM. Vedi README per il rebuild Core ML.
"""

from __future__ import annotations

import asyncio

import numpy as np

from .config import STTConfig
from .logging_setup import get_logger

log = get_logger(__name__)

# Soglia minima sotto la quale l'audio è considerato vuoto/rumore.
_MIN_SAMPLES = 16000 // 4  # ~250 ms a 16 kHz


class STT:
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        # Import lazy: pywhispercpp scarica/compila il modello al primo uso.
        from pywhispercpp.model import Model

        log.info("stt_loading", model=cfg.model)
        self._model = Model(
            model=cfg.model,
            n_threads=cfg.n_threads,
            language=cfg.language,
            print_realtime=False,
            print_progress=False,
        )
        log.info("stt_ready", model=cfg.model)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.shape[0] < _MIN_SAMPLES:
            return ""
        segments = self._model.transcribe(audio)
        return "".join(seg.text for seg in segments).strip()

    async def transcribe(self, audio: np.ndarray) -> str:
        """Trascrive audio float32 mono 16 kHz in testo italiano."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)
