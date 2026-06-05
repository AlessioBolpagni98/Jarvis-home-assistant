"""Smoke test del loop agentico con tool, SENZA microfono (Milestone 3).

Scrivi un prompt: l'LLM decide se chiamare un tool (timer, meteo, web), il
risultato rientra nella conversazione e la risposta finale viene sintetizzata e
riprodotta in streaming. Stampa i tool usati e il time-to-first-audio.

Prova: «Che tempo fa a Roma?», «imposta un timer di 10 secondi per la pasta».
Richiede Ollama in esecuzione + Kokoro scaricato. web_search usa DuckDuckGo
(ddgs, nessuna chiave API). Uso: uv run python scripts/smoke_agent.py
"""

from __future__ import annotations

import asyncio
import time

from jarvis.agent import Agent
from jarvis.audio_io import PlaybackQueue
from jarvis.config import load_config
from jarvis.llm import LLM
from jarvis.sentences import SentenceSplitter
from jarvis.tools import build_default_registry
from jarvis.tts import TTS


async def main() -> None:
    cfg = load_config()
    print("Carico LLM, TTS e tool...")
    llm = LLM(cfg.llm)
    tts = TTS(cfg.tts)
    registry = build_default_registry(cfg.tools)
    agent = Agent(llm, registry, cfg.llm.max_tool_iterations)
    print("Tool disponibili:", [s["function"]["name"] for s in registry.schemas()])

    history = [{"role": "system", "content": cfg.assistant.system_prompt}]
    loop = asyncio.get_running_loop()
    try:
        while True:
            prompt = await loop.run_in_executor(
                None, input, "\n💬 Prompt (INVIO vuoto per uscire): "
            )
            if not prompt.strip():
                break
            history.append({"role": "user", "content": prompt})

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
                    s = await sentences.get()
                    if s is None:
                        break
                    samples, _ = await tts.synthesize(s)
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - t0) * 1000
                    playback.put(samples)
                    n += 1

            async def on_text(text: str) -> None:
                for s in splitter.feed(text):
                    await sentences.put(s)

            task = asyncio.create_task(worker())
            result = await agent.run(history, on_text)
            for s in splitter.flush():
                await sentences.put(s)
            await sentences.put(None)
            await task
            await loop.run_in_executor(None, playback.stop)

            tools = ", ".join(result.tools_used) or "nessuno"
            ttfa = f"{first_audio_ms:.0f} ms" if first_audio_ms else "—"
            print(f"   🛠  tool: {tools} · ⏱ TTFA: {ttfa}")
            print(f"   🤖 {result.text}")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
