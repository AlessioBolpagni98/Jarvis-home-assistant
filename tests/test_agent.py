"""Test d'integrazione del loop agentico (spec §13).

Un LLM mock emette una tool call nota al primo giro, poi una risposta testuale:
si verifica che il tool venga eseguito, che il risultato rientri nella history
come messaggio ``role="tool"`` e che il modello produca la risposta finale.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import pytest

from jarvis.agent import Agent
from jarvis.llm import Delta, Message, TextDelta, ToolCallDelta
from jarvis.tools.registry import ToolRegistry


class FakeLLM:
    """LLM finto: 1° chiamata → tool call; chiamate successive → testo finale.

    Espone una ``stream_chat`` con la stessa firma di ``jarvis.llm.LLM`` così che
    l'``Agent`` non distingua tra reale e mock.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.saw_tool_result = False

    async def stream_chat(
        self, messages: list[Message], tools=None  # noqa: ANN001
    ) -> AsyncIterator[Delta]:
        self.calls += 1
        if self.calls == 1:
            yield ToolCallDelta(id="c1", name="meteo", arguments={"localita": "Roma"})
            return
        # Al secondo giro l'ultimo messaggio deve essere il risultato del tool.
        last = messages[-1]
        if last.get("role") == "tool":
            self.saw_tool_result = True
        yield TextDelta(text="A Roma ci sono 20 gradi. ")
        yield TextDelta(text="Bella giornata.")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(name="meteo", description="Meteo finto per i test.")
    def meteo(localita: str = "") -> str:
        return f"sereno a {localita}"

    return reg


async def test_agent_runs_tool_then_answers() -> None:
    llm = FakeLLM()
    reg = _registry()
    agent = Agent(llm, reg, max_iterations=5)

    spoken: list[str] = []

    async def on_text(text: str) -> None:
        spoken.append(text)

    history: list[Message] = [{"role": "user", "content": "Che tempo fa a Roma?"}]
    result = await agent.run(history, on_text)

    # Il tool è stato eseguito e il suo risultato è rientrato nel loop.
    assert llm.saw_tool_result is True
    assert result.tools_used == ["meteo"]

    # La risposta finale è quella testuale, ricostruita dai delta.
    assert result.text == "A Roma ci sono 20 gradi. Bella giornata."
    assert "".join(spoken) == "A Roma ci sono 20 gradi. Bella giornata."

    # La history contiene: user, assistant(tool_calls), tool, assistant(testo).
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # Il messaggio assistant porta la tool call in formato OpenAI: id + arguments
    # come stringa JSON (così LiteLLM la rigira a qualsiasi provider).
    tool_call = history[1]["tool_calls"][0]
    assert tool_call["id"] == "c1"
    assert tool_call["function"]["name"] == "meteo"
    assert json.loads(tool_call["function"]["arguments"]) == {"localita": "Roma"}

    # Il risultato del tool è collegato alla call via tool_call_id.
    tool_msg = history[2]
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "sereno a Roma"


async def test_agent_calls_on_tool_start_before_executing() -> None:
    """``on_tool_start`` è invocato con nome+argomenti prima del risultato del tool."""
    llm = FakeLLM()
    reg = _registry()
    agent = Agent(llm, reg, max_iterations=5)

    started: list[tuple[str, dict]] = []

    async def on_tool_start(name: str, arguments: dict) -> None:
        # Quando annunciamo il tool, il suo risultato non è ancora rientrato.
        assert llm.saw_tool_result is False
        started.append((name, arguments))

    history: list[Message] = [{"role": "user", "content": "Che tempo fa a Roma?"}]
    await agent.run(history, on_text=lambda _t: _noop(), on_tool_start=on_tool_start)

    assert started == [("meteo", {"localita": "Roma"})]


async def test_agent_without_tools_answers_directly() -> None:
    class PlainLLM:
        async def stream_chat(self, messages, tools=None):  # noqa: ANN001
            yield TextDelta(text="Ciao!")

    agent = Agent(PlainLLM(), ToolRegistry(), max_iterations=3)
    history: list[Message] = [{"role": "user", "content": "Ciao"}]
    result = await agent.run(history, on_text=lambda _t: _noop())
    assert result.text == "Ciao!"
    assert result.tools_used == []


async def _noop() -> None:
    return None
