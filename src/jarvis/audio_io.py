"""Audio I/O basato su sounddevice.

Responsabilità (spec §6.1): catturare audio dal microfono (16 kHz mono, float32)
e riprodurre l'audio del TTS. Espone una cattura "push-to-talk" per la Milestone 1
e una coda di riproduzione per lo streaming TTS (Milestone 2).
"""

from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd

from .config import AudioConfig
from .logging_setup import get_logger

log = get_logger(__name__)


def _device_or_default(name: str) -> str | int | None:
    """Converte una stringa di config in un device sounddevice valido."""
    if not name:
        return None
    return int(name) if name.isdigit() else name


class AudioIO:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._input_device = _device_or_default(cfg.input_device)
        self._output_device = _device_or_default(cfg.output_device)

    @property
    def output_device(self) -> str | int | None:
        """Device di uscita risolto (per la coda di riproduzione in streaming)."""
        return self._output_device

    # ------------------------------------------------------------------ capture
    async def record_push_to_talk(self) -> np.ndarray:
        """Registra finché l'utente non preme INVIO. Ritorna audio float32 mono.

        Pensato per il MVP della Milestone 1: l'utente preme INVIO per iniziare
        (gestito dal chiamante) e di nuovo per terminare.
        """
        frames: list[np.ndarray] = []
        stop = threading.Event()

        def callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
            if status:
                log.warning("audio_input_status", status=str(status))
            frames.append(indata.copy())

        loop = asyncio.get_running_loop()

        def _record() -> np.ndarray:
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="float32",
                blocksize=self.cfg.chunk_samples,
                device=self._input_device,
                callback=callback,
            ):
                input()  # blocca finché l'utente preme INVIO
            stop.set()
            if not frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(frames, axis=0).reshape(-1).astype(np.float32)

        return await loop.run_in_executor(None, _record)

    # ----------------------------------------------------------------- playback
    async def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Riproduce un blocco audio (bloccante, eseguito in executor)."""
        loop = asyncio.get_running_loop()

        def _play() -> None:
            sd.play(audio, samplerate=sample_rate, device=self._output_device)
            sd.wait()

        await loop.run_in_executor(None, _play)


class PlaybackQueue:
    """Coda di riproduzione per lo streaming TTS (Milestone 2).

    Il produttore (orchestratore) accoda blocchi audio man mano che il TTS li
    genera; un thread consumer li riproduce in ordine senza gap percepibili.
    """

    def __init__(self, sample_rate: int, output_device: str | int | None = None) -> None:
        self.sample_rate = sample_rate
        self._output_device = output_device
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()

    def put(self, audio: np.ndarray) -> None:
        self._q.put(audio)

    def _consume(self) -> None:
        while True:
            audio = self._q.get()
            if audio is None:
                break
            sd.play(audio, samplerate=self.sample_rate, device=self._output_device)
            sd.wait()

    def stop(self) -> None:
        """Segnala la fine e attende lo svuotamento della coda."""
        self._q.put(None)
        if self._thread is not None:
            self._thread.join()
