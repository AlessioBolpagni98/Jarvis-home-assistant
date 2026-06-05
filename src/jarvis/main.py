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
import functools
import threading
import time
from datetime import datetime

import numpy as np

from .agent import Agent
from .audio_feedback import Chime, ProcessingSound
from .audio_io import AudioIO, PlaybackQueue
from .config import Config, load_config
from .llm import LLM
from .logging_setup import get_logger, setup_logging, turn_logger
from .sentences import SentenceSplitter
from .stt import STT
from .tools import build_default_registry
from .tts import TTS

log = get_logger("jarvis")

# Nomi italiani di giorni e mesi: evitano la dipendenza dal locale di sistema
# (``locale.setlocale`` è fragile e non thread-safe) per una data leggibile.
_GIORNI = (
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"
)
_MESI = (
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
)


def _temporal_context(now: datetime | None = None) -> str:
    """Sezione di contesto temporale da appendere al system prompt.

    Data e ora correnti in forma esplicita per un LLM: una riga in italiano
    naturale (leggibile/parlabile) più l'ISO 8601 (non ambigua). Ricalcolata a
    ogni turno, così resta valida anche in esecuzione always-on.
    """
    now = now or datetime.now()
    data = f"{_GIORNI[now.weekday()]} {now.day} {_MESI[now.month - 1]} {now.year}"
    return (
        "<CONTESTO_TEMPORALE>\n"
        f"Oggi è {data}, sono le {now:%H:%M}.\n"
        f"Data e ora in formato ISO 8601: {now:%Y-%m-%dT%H:%M:%S}.\n"
        "</CONTESTO_TEMPORALE>"
    )


class Assistant:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.audio = AudioIO(cfg.audio)
        self.processing_sound = ProcessingSound(cfg.audio_feedback, self.audio.output_device)
        self.chime = Chime(cfg.audio_feedback, self.audio.output_device)
        self.stt = STT(cfg.stt)
        self.llm = LLM(cfg.llm)
        self.tts = TTS(cfg.tts)
        self.tools = build_default_registry(cfg.tools)
        self.agent = Agent(self.llm, self.tools, cfg.llm.max_tool_iterations)
        # Prompt base (statico, da config) + contesto temporale, ricalcolato a ogni
        # turno in ``_respond`` così che data e ora restino sempre attuali.
        self.system_prompt = cfg.assistant.system_prompt
        self.history: list[dict] = [
            {"role": "system", "content": self._system_content()}
        ]
        # Listener wake word: inizializzato pigramente solo in modalità wake_word.
        self._wake = None

    def _system_content(self) -> str:
        """System prompt base + sezione data/ora corrente (aggiornata a ogni turno)."""
        return f"{self.system_prompt}\n\n{_temporal_context()}"

    async def _respond(self, user_text: str, t) -> str:  # noqa: ANN001
        """Loop agentico con streaming LLM→TTS.

        Avvia la coda di riproduzione e un worker TTS; passa all'``Agent`` un
        callback che spezza il testo in frasi e le accoda man mano. Registra in
        ``t`` il time-to-first-audio, il numero di frasi e i tool usati.
        """
        # Aggiorna data/ora nel system prompt prima di rispondere (always-on safe).
        self.history[0]["content"] = self._system_content()
        self.history.append({"role": "user", "content": user_text})

        splitter = SentenceSplitter(
            language=self.cfg.stt.language,
            max_chars=self.cfg.tts.max_sentence_chars,
        )

        # Coordinamento earcon↔voce: l'earcon "tastiera" riempie le fasi di attesa
        # (thinking LLM, esecuzione tool + secondo giro) e tace quando parla la voce.
        # ``thinking`` è alzato entrando in attesa e abbassato al primo testo della
        # risposta. ``on_active`` (prima di ogni blocco vocale) spegne l'earcon: così
        # i due non si sovrappongono mai. ``on_drained`` (coda svuotata) lo riarma
        # solo se siamo ancora in attesa — quindi non lampeggia tra le frasi di una
        # risposta, ma copre il silenzio dopo il preambolo mentre gira il tool.
        thinking = threading.Event()

        def on_voice_active() -> None:  # thread consumer del playback
            self.processing_sound.stop()

        def on_voice_drained() -> None:  # thread consumer del playback
            if thinking.is_set():
                self.processing_sound.start()  # ponte sul gap tool / secondo giro

        playback = PlaybackQueue(
            self.tts.sample_rate,
            self.audio.output_device,
            on_active=on_voice_active,
            on_drained=on_voice_drained,
        )
        playback.start()

        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        n_spoken = 0
        t0 = time.perf_counter()
        first_audio_ms: float | None = None

        # Primo giro di thinking: earcon acceso finché non parte la voce.
        thinking.set()
        self.processing_sound.start()

        async def tts_worker() -> None:
            nonlocal n_spoken, first_audio_ms
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                samples, _ = await self.tts.synthesize(sentence)
                if first_audio_ms is None:
                    first_audio_ms = round((time.perf_counter() - t0) * 1000, 1)
                playback.put(samples)
                n_spoken += 1

        async def on_text(text: str) -> None:
            thinking.clear()  # sta arrivando la risposta: non siamo più in attesa
            for sentence in splitter.feed(text):
                await sentences.put(sentence)

        async def on_tool_start(name: str, arguments: dict) -> None:
            # Preambolo contestuale ("Controllo il meteo a Roma…"): maschera la
            # latenza del tool con un feedback semantico, non col solo earcon.
            phrase = self.tools.preamble_for(name, arguments)
            if phrase:
                for sentence in splitter.flush():
                    await sentences.put(sentence)
                await sentences.put(phrase)
            thinking.set()  # dopo il preambolo si torna in attesa (tool + 2° giro)

        worker = asyncio.create_task(tts_worker())
        try:
            result = await self.agent.run(self.history, on_text, on_tool_start)
            for sentence in splitter.flush():
                await sentences.put(sentence)
        finally:
            thinking.clear()
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
            # Chime di conferma allo scatto della wake word ("ti ho sentito").
            listen = functools.partial(self.wake.listen, on_wake=self.chime.play)
            return await loop.run_in_executor(None, listen)
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
