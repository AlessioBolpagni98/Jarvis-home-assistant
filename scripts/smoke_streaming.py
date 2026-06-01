"""Smoke test dello streaming LLM→TTS SENZA microfono (Milestone 2).

Scrivi un prompt: l'LLM risponde in streaming, ogni frase completa viene
sintetizzata e riprodotta mentre il modello continua a generare. Stampa il
time-to-first-audio (TTFA), la metrica percettiva chiave della Milestone 2.

Richiede Ollama in esecuzione e i modelli (Kokoro) scaricati; serve un'uscita
audio. Uso: uv run python scripts/smoke_streaming.py
"""

from __future__ import annotations

import asyncio
import time

from jarvis.audio_io import PlaybackQueue
from jarvis.config import load_config
from jarvis.llm import LLM, TextDelta
from jarvis.sentences import SentenceSplitter
from jarvis.tts import TTS


async def speak(cfg, llm: LLM, tts: TTS, prompt: str) -> None:  # noqa: ANN001
    messages = [
        {"role": "system", "content": cfg.assistant.system_prompt},
        {"role": "user", "content": prompt},
    ]
    splitter = SentenceSplitter(cfg.stt.language, cfg.tts.max_sentence_chars)
    playback = PlaybackQueue(tts.sample_rate)
    playback.start()

    sentences: asyncio.Queue[str | None] = asyncio.Queue()
    t0 = time.perf_counter()
    first_audio_ms: float | None = None
    n = 0

    async def worker() -> None:
        nonlocal first_audio_ms, n
        while True:
            sentence = await sentences.get()
            if sentence is None:
                break
            samples, _ = await tts.synthesize(sentence)
            if first_audio_ms is None:
                first_audio_ms = (time.perf_counter() - t0) * 1000
            playback.put(samples)
            n += 1
            print(f"   🔊 [{n}] {sentence!r}")

    task = asyncio.create_task(worker())
    async for delta in llm.stream_chat(messages):
        if isinstance(delta, TextDelta):
            for sentence in splitter.feed(delta.text):
                await sentences.put(sentence)
    for sentence in splitter.flush():
        await sentences.put(sentence)
    await sentences.put(None)
    await task
    await asyncio.get_running_loop().run_in_executor(None, playback.stop)

    print(f"\n   ⏱  time-to-first-audio: {first_audio_ms:.0f} ms · {n} frasi")


async def main() -> None:
    cfg = load_config()
    print("Carico LLM e TTS...")
    llm = LLM(cfg.llm)
    tts = TTS(cfg.tts)
    loop = asyncio.get_running_loop()
    try:
        while True:
            prompt = await loop.run_in_executor(
                None, input, "\n💬 Prompt (INVIO vuoto per uscire): "
            )
            if not prompt.strip():
                break
            await speak(cfg, llm, tts, prompt)
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
