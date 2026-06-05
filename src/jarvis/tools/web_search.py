"""Tool ``web_search`` — DuckDuckGo via libreria ``ddgs`` (spec §7.2/§7.3).

Scelta concordata: backend **open source senza chiave API**. La libreria ``ddgs``
interroga i motori di ricerca (DuckDuckGo e affini) con ``backend="auto"``: ruota
tra più motori (duckduckgo, bing, brave, google, ...) e fa fallback automatico se
uno è bloccato o rate-limited → più robusto per un assistente vocale.

``ddgs`` è **sincrono**: la chiamata bloccante gira in un thread
(``asyncio.to_thread``) così non blocca l'event loop (earcon di processing, TTS).

L'LLM riceve titolo + snippet dei primi risultati e li sintetizza nella risposta
(privacy: le query escono verso i motori scelti da ddgs, vedi NFR2).
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from ddgs import DDGS
from pydantic import Field

from ..logging_setup import get_logger

log = get_logger(__name__)

_TIMEOUT = 10


def register(registry, n_results: int, region: str) -> None:  # noqa: ANN001
    """Registra il tool ``web_search`` sul registry."""

    @registry.tool(
        name="web_search",
        description=(
            "Cerca sul web informazioni aggiornate o fattuali (notizie, dati, eventi). "
            "Usalo quando la risposta richiede conoscenze che potresti non avere o che "
            "cambiano nel tempo."
        ),
        preamble=[
            "Faccio una ricerca sul web.",
            "Cerco sul web, un momento.",
            "Vado a controllare in rete.",
        ],
    )
    async def web_search(
        query: Annotated[str, Field(description="La query di ricerca, in linguaggio naturale")],
    ) -> str:
        def _search() -> list[dict]:
            return DDGS(timeout=_TIMEOUT).text(
                query, region=region, max_results=n_results, backend="auto"
            )

        results = await asyncio.to_thread(_search)
        if not results:
            return f"Nessun risultato per «{query}»."

        righe = [
            f"- {r.get('title', '').strip()}: {r.get('body', '').strip()}"
            for r in results[:n_results]
        ]
        log.info("web_search_ok", query=query, n=len(righe))
        return f"Risultati per «{query}»:\n" + "\n".join(righe)
