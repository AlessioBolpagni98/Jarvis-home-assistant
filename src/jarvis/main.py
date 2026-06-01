"""Entrypoint — orchestratore del turno (Milestone 1→4).

Giro completo: cattura → Whisper (STT) → loop agentico LLM+tool → Kokoro (TTS) →
altoparlante. Due elementi chiave si combinano qui:

- **Streaming LLM→TTS** (M2): appena l'LLM chiude la **prima frase**, va al TTS e
  la voce parte mentre il modello continua (FR8, NFR1).
- **Loop agentico** (M3): l'LLM può chiamare i tool (timer, meteo, web); i risultati
  rientrano nella conversazione e il modello prosegue (``agent.Agent``).

La **cattura** dipende dalla modalità (``[runtime].mode``):

- ``push_to_talk`` — premi INVIO per parlare (sviluppo/test).
- ``wake_word`` — ascolto continuo di «hey jarvis» + endpointing VAD (M4, always-on).

Il resto del turno (STT → risposta) è identico nelle due modalità.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from .agent import Agent
from .audio_feedback import ProcessingSound
from .audio_io import AudioIO, PlaybackQueue
from .config import Config, load_config
from .llm import LLM
from .logging_setup import get_logger, setup_logging, turn_logger
from .sentences import SentenceSplitter
from .stt import STT
from .tools import build_default_registry
from .tts import TTS

log = get_logger("jarvis")


class Assistant:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.audio = AudioIO(cfg.audio)
        self.processing_sound = ProcessingSound(cfg.audio_feedback, self.audio.output_device)
        self.stt = STT(cfg.stt)
        self.llm = LLM(cfg.llm)
        self.tts = TTS(cfg.tts)
        self.tools = build_default_registry(cfg.tools)
        self.agent = Agent(self.llm, self.tools, cfg.llm.max_tool_iterations)
        self.history: list[dict] = [
            {"role": "system", "content": cfg.assistant.system_prompt}
        ]
        # Listener wake word: inizializzato pigramente solo in modalità wake_word.
        self._wake = None

    async def _respond(self, user_text: str, t) -> str:  # noqa: ANN001
        """Loop agentico con streaming LLM→TTS.

        Avvia la coda di riproduzione e un worker TTS; passa all'``Agent`` un
        callback che spezza il testo in frasi e le accoda man mano. Registra in
        ``t`` il time-to-first-audio, il numero di frasi e i tool usati.
        """
        self.history.append({"role": "user", "content": user_text})

        # Earcon di processing: parte ora (inizio del thinking LLM) e si spegne
        # appena il TTS produce il primo audio, lasciando il posto alla voce.
        self.processing_sound.start()

        splitter = SentenceSplitter(
            language=self.cfg.stt.language,
            max_chars=self.cfg.tts.max_sentence_chars,
        )
        playback = PlaybackQueue(self.tts.sample_rate, self.audio.output_device)
        playback.start()

        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        n_spoken = 0
        t0 = time.perf_counter()
        first_audio_ms: float | None = None

        async def tts_worker() -> None:
            nonlocal n_spoken, first_audio_ms
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                samples, _ = await self.tts.synthesize(sentence)
                if first_audio_ms is None:
                    # Primo audio pronto: spegni la tastiera prima di dar voce.
                    self.processing_sound.stop()
                    first_audio_ms = round((time.perf_counter() - t0) * 1000, 1)
                playback.put(samples)
                n_spoken += 1

        async def on_text(text: str) -> None:
            for sentence in splitter.feed(text):
                await sentences.put(sentence)

        worker = asyncio.create_task(tts_worker())
        try:
            result = await self.agent.run(self.history, on_text)
            for sentence in splitter.flush():
                await sentences.put(sentence)
        finally:
            self.processing_sound.stop()  # rete di sicurezza: nessun audio / errore
            await sentences.put(None)  # sentinella: chiude il worker
            await worker
            await asyncio.get_running_loop().run_in_executor(None, playback.stop)

        t.set(ttfa_ms=first_audio_ms, n_sentences=n_spoken, tools=result.tools_used)
        return result.text

    @property
    def wake(self):  # noqa: ANN201
        """Listener wake word, costruito alla prima richiesta (carica i modelli)."""
        if self._wake is None:
            from .wake_word import WakeWordListener

            self._wake = WakeWordListener(self.cfg.audio, self.cfg.wakeword, self.cfg.vad)
        return self._wake

    async def _capture(self) -> np.ndarray:
        """Cattura un'utterance secondo la modalità configurata."""
        loop = asyncio.get_running_loop()
        if self.cfg.runtime.mode == "wake_word":
            print("\n👂 In ascolto di «hey jarvis»...")
            return await loop.run_in_executor(None, self.wake.listen)
        # push_to_talk
        await loop.run_in_executor(None, input, "\n⏎ Premi INVIO per parlare...")
        print("🎙️  Sto ascoltando... premi INVIO per terminare.")
        return await self.audio.record_push_to_talk()

    async def turn(self) -> None:
        with turn_logger(log) as t:
            with t.stage("capture"):
                audio = await self._capture()

            with t.stage("stt"):
                user_text = await self.stt.transcribe(audio)
            t.set(stt_text=user_text)

            if not user_text:
                print("…non ho sentito nulla.")
                return
            print(f"👤 {user_text}")

            with t.stage("respond"):
                answer = await self._respond(user_text, t)
            print(f"🤖 {answer}")

    async def run(self) -> None:
        print(f"Jarvis pronto (modalità: {self.cfg.runtime.mode}). Ctrl-C per uscire.")
        try:
            while True:
                try:
                    await self.turn()
                except (EOFError, KeyboardInterrupt):
                    print("\nA presto!")
                    return
                except Exception:  # noqa: BLE001
                    log.exception("turn_error")
        finally:
            await self.llm.aclose()


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)
    assistant = Assistant(cfg)
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
