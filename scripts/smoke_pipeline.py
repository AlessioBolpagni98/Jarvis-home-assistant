"""Smoke test della pipeline SENZA microfono (Milestone 1).

Round-trip: Kokoro (TTS) genera una frase italiana → la si ri-trascrive con
Whisper (STT) → si manda una domanda all'LLM. Valida i tre stadi end-to-end
senza hardware audio. Richiede i modelli scaricati e Ollama in esecuzione.

Uso: uv run python scripts/smoke_pipeline.py
"""

from __future__ import annotations

import asyncio
import time

import httpx
import numpy as np

from jarvis.config import load_config
from jarvis.stt import STT
from jarvis.tts import TTS


async def llm_answer_native(cfg, prompt: str) -> str:
    """Risposta LLM via endpoint nativo Ollama con think disabilitato."""
    base = cfg.llm.base_url.removesuffix("/v1")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{base}/api/chat",
            json={
                "model": cfg.llm.model,
                "messages": [
                    {"role": "system", "content": cfg.assistant.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"num_predict": cfg.llm.max_tokens},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


async def main() -> None:
    cfg = load_config()

    print("\n[1/3] TTS — sintetizzo una frase italiana con Kokoro...")
    tts = TTS(cfg.tts)
    frase = "Ciao, sono Jarvis. Che tempo fa oggi a Milano?"
    t0 = time.perf_counter()
    samples, sr = await tts.synthesize(frase)
    print(f"      ok: {len(samples)} campioni @ {sr} Hz in {time.perf_counter()-t0:.2f}s")

    # Whisper si aspetta 16 kHz: ricampiono linearmente l'uscita di Kokoro (24 kHz).
    target_sr = cfg.audio.sample_rate
    n_target = int(len(samples) * target_sr / sr)
    resampled = np.interp(
        np.linspace(0, len(samples), n_target, endpoint=False),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)

    print("\n[2/3] STT — ri-trascrivo l'audio con Whisper (primo avvio: scarica il modello)...")
    stt = STT(cfg.stt)
    t0 = time.perf_counter()
    testo = await stt.transcribe(resampled)
    print(f"      frase originale : {frase!r}")
    print(f"      trascrizione    : {testo!r}")
    print(f"      ({time.perf_counter()-t0:.2f}s)")

    print("\n[3/3] LLM — domanda all'LLM (think disabilitato)...")
    t0 = time.perf_counter()
    risposta = await llm_answer_native(cfg, "In una frase: chi sei e cosa sai fare?")
    print(f"      risposta: {risposta!r}")
    print(f"      ({time.perf_counter()-t0:.2f}s)")

    print("\n✅ Pipeline TTS → STT → LLM verificata.")


if __name__ == "__main__":
    asyncio.run(main())
