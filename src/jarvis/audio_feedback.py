"""Feedback audio di processing — earcon "sto pensando" (latenza percepita).

Abilitando il *thinking* dell'LLM il tool calling è più efficace ma la latenza tra
domanda e risposta cresce. Per non lasciare l'utente nel silenzio, appena parte il
turno agentico si avvia in loop un suono di tastiera che simula l'elaborazione; il
suono si spegne nell'istante in cui inizia la voce del TTS. La latenza reale è
identica, ma la **percezione** è di un sistema reattivo.

Riproduzione gapless via ``sd.OutputStream`` con callback su un buffer pre-decodificato,
**separata** dalla ``PlaybackQueue`` del TTS (che usa il ``sd.play`` modulo-level): i
due stream non condividono stato e quindi non si calpestano. Il caricamento è tollerante
ai guasti — se il file manca o non si decodifica, ``start``/``stop`` diventano no-op e il
turno prosegue senza voce di processing.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import AudioFeedbackConfig
from .logging_setup import get_logger

log = get_logger(__name__)


class ProcessingSound:
    """Suono di processing in loop, non bloccante, con stop immediato.

    Il file viene decodificato una sola volta all'avvio. ``start()`` apre uno stream
    di uscita che riproduce il buffer in loop continuo; ``stop()`` lo chiude. Entrambi
    sono idempotenti e thread-safe.
    """

    def __init__(self, cfg: AudioFeedbackConfig, output_device: str | int | None = None) -> None:
        self.cfg = cfg
        self._output_device = output_device
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._pos = 0
        self._data: np.ndarray | None = None
        self._sample_rate = 0
        self._channels = 0

        if not cfg.enabled:
            return
        self._load(cfg.processing_sound, cfg.volume)

    # ------------------------------------------------------------------ loading
    def _load(self, path: str, volume: float) -> None:
        """Decodifica il file (una volta) in float32; degrada a no-op se fallisce."""
        try:
            data, sample_rate = sf.read(str(Path(path)), dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001 — il feedback non deve mai rompere il turno
            log.warning("processing_sound_load_failed", path=path)
            return
        self._data = np.ascontiguousarray(data * float(volume), dtype=np.float32)
        self._sample_rate = int(sample_rate)
        self._channels = self._data.shape[1]
        log.info(
            "processing_sound_ready",
            path=path,
            sample_rate=self._sample_rate,
            channels=self._channels,
        )

    # ------------------------------------------------------------------ playback
    def _callback(self, outdata: np.ndarray, frames: int, _time, _status) -> None:  # noqa: ANN001
        """Riempie ``outdata`` leggendo il buffer in loop (wrap-around)."""
        data = self._data
        assert data is not None  # garantito: lo stream esiste solo se _data è valido
        n = len(data)
        pos = self._pos
        written = 0
        while written < frames:
            take = min(n - pos, frames - written)
            outdata[written : written + take] = data[pos : pos + take]
            written += take
            pos += take
            if pos >= n:
                pos = 0
        self._pos = pos

    def start(self) -> None:
        """Avvia il loop del suono di processing (no-op se già attivo o disabilitato)."""
        if self._data is None:
            return
        with self._lock:
            if self._stream is not None:
                return
            self._pos = 0
            try:
                self._stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="float32",
                    device=self._output_device,
                    callback=self._callback,
                )
                self._stream.start()
            except Exception:  # noqa: BLE001
                log.warning("processing_sound_start_failed")
                self._stream = None

    def stop(self) -> None:
        """Ferma e chiude lo stream (idempotente)."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                log.warning("processing_sound_stop_failed")


class Chime:
    """Earcon one-shot, non bloccante — conferma "ti ho sentito" alla wake word.

    Quando «hey jarvis» scatta, l'utente non ha alcun riscontro udibile che il
    sistema l'abbia colto (il dubbio "mi ha sentito?" è la classica frizione degli
    assistenti vocali). Un suono brevissimo allo scatto lo elimina, come il *ding*
    di Siri/Alexa. Riprodotto fire-and-forget così da non ritardare la cattura.

    Tollerante ai guasti come ``ProcessingSound``: se il file manca o non si
    decodifica (es. non su macOS, dove il default è un suono di sistema), ``play``
    diventa un no-op e l'ascolto prosegue senza chime.
    """

    def __init__(self, cfg: AudioFeedbackConfig, output_device: str | int | None = None) -> None:
        self._output_device = output_device
        self._data: np.ndarray | None = None
        self._sample_rate = 0

        if not cfg.wake_chime_enabled:
            return
        try:
            data, sample_rate = sf.read(
                str(Path(cfg.wake_chime)), dtype="float32", always_2d=True
            )
        except Exception:  # noqa: BLE001 — il chime non deve mai bloccare l'ascolto
            log.warning("wake_chime_load_failed", path=cfg.wake_chime)
            return
        self._data = np.ascontiguousarray(
            data * float(cfg.wake_chime_volume), dtype=np.float32
        )
        self._sample_rate = int(sample_rate)
        log.info("wake_chime_ready", path=cfg.wake_chime, sample_rate=self._sample_rate)

    def play(self) -> None:
        """Riproduce il chime (non bloccante). No-op se non caricato/abilitato."""
        if self._data is None:
            return
        try:
            sd.play(self._data, samplerate=self._sample_rate, device=self._output_device)
        except Exception:  # noqa: BLE001
            log.warning("wake_chime_play_failed")
