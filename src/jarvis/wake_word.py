"""Wake word + endpointing VAD (spec §6.2, Milestone 4).

Ascolto continuo a bassa risorsa sulla **CPU**: openWakeWord rileva «hey jarvis»;
appena scatta, Silero VAD chiude la cattura al termine del parlato (endpointing).
Ritorna il buffer dell'utterance, pronto per lo STT.

Un'unica finestra audio da **512 campioni a 16 kHz** (32 ms) serve entrambi: è la
dimensione richiesta da Silero VAD, e openWakeWord accetta chunk di qualsiasi
lunghezza (bufferizza internamente). Entrambi i modelli sono ONNX (onnxruntime),
così non si contende la GPU dell'LLM né il Neural Engine dello STT.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import sounddevice as sd

from .audio_io import _device_or_default
from .config import AudioConfig, VADConfig, WakeWordConfig
from .logging_setup import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16000
_WINDOW = 512  # campioni richiesti da Silero VAD a 16 kHz (32 ms)


class WakeWordListener:
    """Rileva la wake word e cattura l'utterance successiva con endpointing VAD."""

    def __init__(
        self, audio: AudioConfig, wakeword: WakeWordConfig, vad: VADConfig
    ) -> None:
        import openwakeword
        from openwakeword.model import Model
        from silero_vad import load_silero_vad

        self._vad_cfg = vad
        self._threshold = wakeword.threshold
        self._device = _device_or_default(audio.input_device)

        # Risolve il modello pre-addestrato per nome (es. "hey_jarvis").
        paths = openwakeword.get_pretrained_model_paths()
        match = [p for p in paths if wakeword.model in os.path.basename(p)]
        if not match:
            available = sorted(os.path.basename(p) for p in paths)
            raise FileNotFoundError(
                f"wake word {wakeword.model!r} non trovata. Disponibili: {available}"
            )
        model_path = match[0]
        # La chiave nello score di predict() è il nome file senza estensione.
        self._key = os.path.basename(model_path).removesuffix(".onnx")

        log.info("wakeword_loading", model=self._key)
        self._oww = Model(wakeword_model_paths=[model_path])
        self._silero = load_silero_vad(onnx=True)
        log.info("wakeword_ready", model=self._key, threshold=self._threshold)

    def listen(self, on_wake: Callable[[], None] | None = None) -> np.ndarray:
        """Blocca fino alla wake word, poi cattura l'utterance (bloccante).

        Pensato per essere eseguito in un executor. Ritorna audio float32 mono a
        16 kHz; array vuoto se scade il timeout senza parlato. ``on_wake`` (se
        fornito) viene chiamato nell'istante in cui la wake word scatta — utile per
        un chime di conferma; eventuali errori sono ignorati per non perdere la cattura.
        """
        from silero_vad import VADIterator

        self._oww.reset()
        vad = VADIterator(
            self._silero,
            sampling_rate=_SAMPLE_RATE,
            min_silence_duration_ms=self._vad_cfg.silence_ms,
        )

        captured: list[np.ndarray] = []
        speaking = False
        capturing = False
        frames = 0
        max_frames = int(self._vad_cfg.max_capture_s * _SAMPLE_RATE / _WINDOW)

        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_WINDOW,
            device=self._device,
        ) as stream:
            while True:
                block, _ = stream.read(_WINDOW)
                pcm = block.reshape(-1)

                if not capturing:
                    score = float(self._oww.predict(pcm).get(self._key, 0.0))
                    if score >= self._threshold:
                        log.info("wake_detected", score=round(score, 3))
                        if on_wake is not None:
                            try:
                                on_wake()
                            except Exception:  # noqa: BLE001
                                log.warning("wake_chime_failed")
                        capturing = True
                        vad.reset_states()
                    continue

                # Fase di cattura: accumula e cerca il fine-frase col VAD.
                f32 = pcm.astype(np.float32) / 32768.0
                captured.append(f32)
                frames += 1

                event = vad(f32)
                if event and "start" in event:
                    speaking = True
                if event and "end" in event and speaking:
                    break
                if frames >= max_frames:
                    log.info("capture_timeout", seconds=self._vad_cfg.max_capture_s)
                    break

        if not speaking or not captured:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(captured).astype(np.float32)
