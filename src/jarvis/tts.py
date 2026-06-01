"""TTS — Kokoro-82M via kokoro-onnx (spec §6.5).

Contratto: ``synthesize(text) -> (audio, sample_rate)`` con sintesi **per frase**.
Gira su CPU (onnxruntime), footprint trascurabile. Per l'italiano serve il codice
lingua corretto (``lang="i"``) per un g2p adeguato.
"""

from __future__ import annotations

import asyncio

import numpy as np

from .config import TTSConfig
from .logging_setup import get_logger

log = get_logger(__name__)


class TTS:
    # Kokoro v1.0 sintetizza sempre a 24 kHz: utile a monte per dimensionare la
    # coda di riproduzione in streaming senza attendere il primo chunk.
    SAMPLE_RATE = 24000

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self.sample_rate = self.SAMPLE_RATE
        from kokoro_onnx import Kokoro

        log.info("tts_loading", model=cfg.model_path, voice=cfg.voice)
        self._kokoro = Kokoro(cfg.model_path, cfg.voices_path)
        log.info("tts_ready", voice=cfg.voice)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        samples, sample_rate = self._kokoro.create(
            text,
            voice=self.cfg.voice,
            speed=self.cfg.speed,
            lang=self.cfg.lang,
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Sintetizza ``text`` in audio. Ritorna (campioni float32, sample_rate)."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), 24000
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)
