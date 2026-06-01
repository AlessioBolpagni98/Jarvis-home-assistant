"""Tool ``web_search`` — Brave Search API (spec §7.2/§7.3).

Scelta concordata: backend **Brave** (opzione A). La chiave NON sta in config: si
passa via ``JARVIS_TOOLS__BRAVE_API_KEY``. Se manca, il tool fallisce con un
messaggio chiaro che l'LLM può riferire all'utente.

L'LLM riceve titolo + snippet dei primi risultati e li sintetizza nella risposta
(privacy: le query escono verso Brave, vedi NFR2).
"""

from __future__ import annotations

from typing import Annotated

import httpx
from pydantic import Field

from ..logging_setup import get_logger

log = get_logger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT = 10.0


def register(registry, api_key: str, n_results: int) -> None:  # noqa: ANN001
    """Registra il tool ``web_search`` sul registry."""

    @registry.tool(
        name="web_search",
        description=(
            "Cerca sul web informazioni aggiornate o fattuali (notizie, dati, eventi). "
            "Usalo quando la risposta richiede conoscenze che potresti non avere o che "
            "cambiano nel tempo."
        ),
    )
    async def web_search(
        query: Annotated[str, Field(description="La query di ricerca, in linguaggio naturale")],
    ) -> str:
        if not api_key:
            raise RuntimeError(
                "ricerca web non configurata: manca JARVIS_TOOLS__BRAVE_API_KEY"
            )
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                _BRAVE_URL,
                params={"q": query, "count": n_results, "search_lang": "it"},
                headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            )
            resp.raise_for_status()
            results = (resp.json().get("web") or {}).get("results") or []

        if not results:
            return f"Nessun risultato per «{query}»."

        righe = []
        for r in results[:n_results]:
            titolo = r.get("title", "").strip()
            descr = r.get("description", "").strip()
            righe.append(f"- {titolo}: {descr}")
        log.info("web_search_ok", query=query, n=len(righe))
        return f"Risultati per «{query}»:\n" + "\n".join(righe)
