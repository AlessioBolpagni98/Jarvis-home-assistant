"""LLM — backend astratto dietro un'interfaccia unica (spec §6.4).

Contratto:

    stream_chat(messages, tools) -> AsyncIterator[Delta]

dove ``Delta`` è o un frammento di **testo** (``TextDelta``) o una **richiesta di
tool call** completa (``ToolCallDelta``). Incapsula il backend così da poter
cambiare runtime senza toccare l'orchestratore.

Implementazione: **LiteLLM** (``litellm.acompletion``), interfaccia unica verso
100+ provider. Il provider è scelto dal **prefisso del modello** in config
(es. ``ollama_chat/qwen3...`` per l'Ollama locale, ``openai/gpt-4o`` per il cloud):
cambiare runtime significa cambiare una riga di config, senza toccare questo modulo.

Due dettagli di LiteLLM gestiti qui:

- ``think: false`` (disabilita il ragionamento — essenziale per un assistente
  vocale) è passato come parametro **top-level** via ``extra_params`` ed è onorato
  da Ollama solo col prefisso ``ollama_chat/`` (→ ``/api/chat``).
- in streaming le tool call arrivano **frammentate** su più chunk (formato OpenAI:
  ``delta.tool_calls`` con ``index`` e ``arguments`` parziali): vanno riassemblate
  per ``index`` ed emesse complete a fine stream.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import litellm

from .config import LLMConfig
from .logging_setup import get_logger

log = get_logger(__name__)

# LiteLLM è molto verboso di default: silenziamo il rumore non essenziale.
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Un messaggio in formato chat (role/content; per l'assistente anche tool_calls).
Message = dict[str, Any]
# Schema di un tool in formato OpenAI ({"type": "function", "function": {...}}).
ToolSchema = dict[str, Any]


@dataclass
class TextDelta:
    """Un frammento di testo della risposta."""

    text: str


@dataclass
class ToolCallDelta:
    """Una richiesta di tool call completa, pronta per essere eseguita."""

    id: str
    name: str
    arguments: dict[str, Any]


Delta = TextDelta | ToolCallDelta


class LLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    async def aclose(self) -> None:
        # LiteLLM gestisce internamente i propri client HTTP: niente da chiudere.
        return None

    def _kwargs(
        self, messages: list[Message], tools: list[ToolSchema] | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "timeout": self.cfg.request_timeout,
            "drop_params": self.cfg.drop_params,
        }
        if self.cfg.api_base:
            kwargs["api_base"] = self.cfg.api_base
        if self.cfg.api_key:
            kwargs["api_key"] = self.cfg.api_key
        if tools:
            kwargs["tools"] = tools
        # Parametri provider-specific top-level (es. think=false per Ollama).
        kwargs.update(self.cfg.extra_params)
        return kwargs

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[Delta]:
        """Esegue una chiamata in streaming, emettendo testo e tool call.

        Il testo arriva incrementale in ``delta.content`` e viene inoltrato subito.
        Le tool call arrivano a frammenti (``delta.tool_calls``): le si accumula per
        ``index`` e si emette un ``ToolCallDelta`` completo a fine stream.
        """
        response = await litellm.acompletion(**self._kwargs(messages, tools))

        # Accumulo dei frammenti di tool call, indicizzato per posizione nel chunk.
        acc: dict[int, dict[str, Any]] = {}

        async for chunk in response:
            choices = chunk.choices or []
            if not choices:
                continue
            delta = choices[0].delta

            content = getattr(delta, "content", None)
            if content:
                yield TextDelta(text=content)

            for tc in getattr(delta, "tool_calls", None) or []:
                slot = acc.setdefault(
                    tc.index, {"id": None, "name": "", "args": ""}
                )
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] = fn.name
                    if fn.arguments:
                        slot["args"] += fn.arguments

        # A fine stream: deserializza gli argomenti e emette le tool call complete.
        for index in sorted(acc):
            slot = acc[index]
            raw = slot["args"]
            try:
                arguments = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                log.warning("tool_call_bad_json", raw=raw)
                arguments = {}
            yield ToolCallDelta(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                name=slot["name"],
                arguments=arguments,
            )
