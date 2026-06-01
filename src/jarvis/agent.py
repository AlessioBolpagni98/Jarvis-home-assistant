"""Loop agentico — il "cervello" del turno (spec §3.2, §8 punto 4).

Cuce LLM e tool registry nel loop di tool calling: chiama ``stream_chat`` con gli
schemi dei tool; mentre arrivano i ``TextDelta`` li inoltra (callback ``on_text``,
che a monte alimenta lo streaming verso il TTS); se arrivano ``ToolCallDelta``,
esegue gli handler dal registry, appende i risultati alla history e **ri-chiama**
l'LLM. Ripete finché il modello produce una risposta senza tool call, con un
guard-rail sul numero massimo di iterazioni.

Disaccoppiato dall'I/O audio (riceve un callback ``on_text``) così da poter essere
testato con un LLM mock — l'integrazione richiesta dalla spec §13.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from .llm import LLM, Message, TextDelta, ToolCallDelta
from .logging_setup import get_logger
from .tools import ToolRegistry

log = get_logger(__name__)

OnText = Callable[[str], Awaitable[None]]


@dataclass
class TurnResult:
    """Esito del turno agentico: testo completo + tool usati (per il log)."""

    text: str
    tools_used: list[str]


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry, max_iterations: int) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations

    async def run(self, history: list[Message], on_text: OnText) -> TurnResult:
        """Esegue il loop agentico mutando ``history`` in-place.

        ``on_text`` riceve ogni frammento di testo della risposta finale man mano
        che arriva (per lo streaming verso il TTS). Ritorna testo e tool usati.
        """
        schemas = self.tools.schemas() or None
        tools_used: list[str] = []
        full: list[str] = []

        for _ in range(self.max_iterations):
            text_parts: list[str] = []
            tool_calls: list[ToolCallDelta] = []

            async for delta in self.llm.stream_chat(history, schemas):
                if isinstance(delta, TextDelta):
                    text_parts.append(delta.text)
                    full.append(delta.text)
                    await on_text(delta.text)
                elif isinstance(delta, ToolCallDelta):
                    tool_calls.append(delta)

            # Messaggio dell'assistente per questa iterazione (testo + eventuali call).
            # Formato OpenAI canonico (lingua franca di LiteLLM): le tool call portano
            # ``id`` e ``arguments`` come stringa JSON.
            assistant: Message = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
            }
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            history.append(assistant)

            if not tool_calls:
                break  # risposta finale: il loop termina

            # Esegue i tool e re-inietta i risultati come messaggi role="tool".
            for tc in tool_calls:
                tools_used.append(tc.name)
                result = await self.tools.execute(tc.name, tc.arguments)
                content = result.content if result.ok else f"Errore: {result.error}"
                history.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": content}
                )
        else:
            log.warning("agent_max_iterations", limit=self.max_iterations)

        return TurnResult(text="".join(full).strip(), tools_used=tools_used)
