"""Tool registry e contratto dei tool (spec §7).

Il nucleo dell'**estensibilità** (FR6): aggiungere un tool = scrivere una funzione
e registrarla con ``@registry.tool(...)``. Non si tocca l'orchestratore.

Scelte concordate: registrazione via **decoratore** + validazione parametri con
**Pydantic**. Lo schema JSON che l'LLM riceve è derivato automaticamente dalla
firma della funzione (annotazioni di tipo + ``Field(description=...)``), così la
descrizione dei parametri vive accanto al codice.

Contratto:

- l'handler è una funzione (sync o async) con parametri tipizzati; ritorna una
  **stringa** (il contenuto mostrato all'LLM). Qualsiasi eccezione viene catturata
  e trasformata in un ``ToolResult`` di errore: scrivere un tool resta banale.
- ``ToolResult`` (``ok``/``content``/``error``) è ciò che l'orchestratore riceve
  e re-inietta nella conversazione come messaggio ``role="tool"``.
"""

from __future__ import annotations

import inspect
import random
import typing
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError, create_model

from ..logging_setup import get_logger

log = get_logger(__name__)

Handler = Callable[..., str | Awaitable[str]]
# Preambolo vocale opzionale di un tool: una frase pronunciata appena il modello
# decide la call, prima di eseguirla (maschera la latenza con feedback semantico).
# Può essere una stringa fissa, una lista di varianti (scelta a caso, per non
# suonare robotico) o un callable che lo costruisce dagli argomenti della call
# (preambolo contestuale, es. "Controllo il meteo a Roma").
Preamble = str | list[str] | Callable[[dict[str, Any]], str]


@dataclass
class ToolResult:
    """Esito dell'esecuzione di un tool (spec §7.1)."""

    ok: bool
    content: str
    error: str | None = None


@dataclass
class Tool:
    """Un tool registrato: metadati + modello dei parametri + handler."""

    name: str
    description: str
    parameters: type[BaseModel]  # modello Pydantic derivato dalla firma
    handler: Handler
    preamble: Preamble | None = None  # frase detta prima di eseguire (opzionale)

    def schema(self) -> dict[str, Any]:
        """Schema in formato OpenAI/Ollama che l'LLM riceve per decidere."""
        params = self.parameters.model_json_schema()
        params.pop("title", None)  # rumore: il nome è già in function.name
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


def _params_model(name: str, func: Handler) -> type[BaseModel]:
    """Costruisce un modello Pydantic dalla firma della funzione.

    Ogni parametro diventa un campo; le annotazioni ``Annotated[T, Field(...)]``
    portano descrizione e vincoli direttamente nello schema JSON. I parametri
    senza default diventano obbligatori.

    Usa ``get_type_hints`` (non ``signature().annotation``) per risolvere le
    annotazioni quando i moduli usano ``from __future__ import annotations``, che
    le trasforma in stringhe.
    """
    hints = typing.get_type_hints(func, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, p in inspect.signature(func).parameters.items():
        if pname in ("self", "cls"):
            continue
        annotation = hints.get(pname, str)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (annotation, default)
    return create_model(f"{name}_Params", **fields)


class ToolRegistry:
    """Collezione di tool con registrazione, introspezione ed esecuzione."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        description: str,
        preamble: Preamble | None = None,
    ) -> Callable[[Handler], Handler]:
        """Decoratore di registrazione. ``description`` è in italiano (guida l'LLM).

        ``preamble`` (opzionale) è la frase che Jarvis pronuncia appena decide di
        usare questo tool, prima di eseguirlo: vive accanto al tool, così aggiungere
        un feedback vocale resta un'operazione locale (FR6).
        """

        def decorator(func: Handler) -> Handler:
            tool_name = name or func.__name__
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=description,
                parameters=_params_model(tool_name, func),
                handler=func,
                preamble=preamble,
            )
            log.info("tool_registered", tool=tool_name)
            return func

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Schemi di tutti i tool, da passare a ``stream_chat``."""
        return [t.schema() for t in self._tools.values()]

    def preamble_for(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Frase da pronunciare prima di eseguire ``name`` (None se non definita).

        Risolve il ``preamble`` del tool: callable → costruito dagli argomenti
        (preambolo contestuale); lista → variante a caso; stringa → così com'è.
        Fail-safe: qualsiasi errore degrada a ``None`` e il turno prosegue muto.
        """
        tool = self._tools.get(name)
        if tool is None or tool.preamble is None:
            return None
        try:
            preamble = tool.preamble
            if callable(preamble):
                return preamble(arguments)
            if isinstance(preamble, list):
                return random.choice(preamble) if preamble else None
            return preamble
        except Exception:  # noqa: BLE001 — il feedback non deve rompere il turno
            log.warning("preamble_failed", tool=name)
            return None

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Risolve il tool, valida gli argomenti ed esegue l'handler.

        Errori (tool sconosciuto, argomenti non validi, eccezioni nell'handler)
        diventano ``ToolResult(ok=False, ...)``: il loop agentico resta vivo e il
        modello può reagire all'errore.
        """
        tool = self._tools.get(name)
        if tool is None:
            log.warning("tool_unknown", tool=name)
            return ToolResult(False, "", f"tool sconosciuto: {name!r}")

        try:
            validated = tool.parameters(**arguments)
        except ValidationError as e:
            log.warning("tool_bad_args", tool=name, error=str(e))
            return ToolResult(False, "", f"argomenti non validi per {name!r}: {e}")

        try:
            result = tool.handler(**validated.model_dump())
            if inspect.isawaitable(result):
                result = await result
            content = str(result)
            log.info("tool_executed", tool=name)
            return ToolResult(True, content)
        except Exception as e:  # noqa: BLE001 — l'errore torna all'LLM, non crasha il turno
            log.warning("tool_failed", tool=name, error=str(e))
            return ToolResult(False, "", f"errore eseguendo {name!r}: {e}")
