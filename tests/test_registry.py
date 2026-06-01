"""Test del tool registry (spec §13): registrazione, schema, esecuzione, errori.

Niente rete né modelli: i tool sono funzioni pure registrate al volo.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import Field

from jarvis.tools.registry import ToolRegistry


def _registry_with_echo() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(name="somma", description="Somma due numeri interi.")
    def somma(
        a: Annotated[int, Field(description="primo addendo")],
        b: Annotated[int, Field(description="secondo addendo")] = 0,
    ) -> str:
        return str(a + b)

    return reg


def test_schema_includes_name_description_and_params() -> None:
    reg = _registry_with_echo()
    (schema,) = reg.schemas()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "somma"
    assert "Somma due numeri" in fn["description"]
    props = fn["parameters"]["properties"]
    assert set(props) == {"a", "b"}
    assert props["a"]["description"] == "primo addendo"
    # 'a' è obbligatorio (niente default), 'b' no.
    assert fn["parameters"]["required"] == ["a"]


async def test_execute_success_wraps_str_in_toolresult() -> None:
    reg = _registry_with_echo()
    res = await reg.execute("somma", {"a": 2, "b": 3})
    assert res.ok is True
    assert res.content == "5"
    assert res.error is None


async def test_execute_uses_default_when_arg_missing() -> None:
    reg = _registry_with_echo()
    res = await reg.execute("somma", {"a": 7})
    assert res.ok is True and res.content == "7"


async def test_unknown_tool_is_error_not_exception() -> None:
    reg = _registry_with_echo()
    res = await reg.execute("inesistente", {})
    assert res.ok is False
    assert "sconosciuto" in (res.error or "")


async def test_invalid_arguments_become_error() -> None:
    reg = _registry_with_echo()
    res = await reg.execute("somma", {"a": "non-un-numero"})
    assert res.ok is False
    assert "non validi" in (res.error or "")


async def test_handler_exception_is_captured() -> None:
    reg = ToolRegistry()

    @reg.tool(name="boom", description="Solleva sempre un errore.")
    def boom() -> str:
        raise ValueError("kaboom")

    res = await reg.execute("boom", {})
    assert res.ok is False
    assert "kaboom" in (res.error or "")


async def test_async_handler_is_awaited() -> None:
    reg = ToolRegistry()

    @reg.tool(name="async_eco", description="Eco asincrona.")
    async def async_eco(testo: str) -> str:
        return testo.upper()

    res = await reg.execute("async_eco", {"testo": "ciao"})
    assert res.ok is True and res.content == "CIAO"
