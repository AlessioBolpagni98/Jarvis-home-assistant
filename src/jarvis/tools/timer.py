"""Tool ``timer`` — locale, nessuna rete (spec §7.2).

Scelte concordate: timer **in memoria** (non sopravvivono al riavvio) + **notifica
macOS** alla scadenza (banner via ``osascript`` e suono di sistema via ``afplay``).

I timer girano come task asyncio in background: scadono indipendentemente dal turno
in corso, quindi la notifica può arrivare mentre Jarvis ascolta o parla d'altro.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Annotated

from pydantic import Field

from ..logging_setup import get_logger

log = get_logger(__name__)

# Suono di sistema macOS riprodotto alla scadenza.
_ALERT_SOUND = "/System/Library/Sounds/Glass.aiff"


def _format_duration(seconds: int) -> str:
    """Durata in italiano leggibile, es. 90 -> '1 minuto e 30 secondi'."""
    minutes, secs = divmod(int(seconds), 60)
    parts: list[str] = []
    if minutes:
        parts.append(f"{minutes} minuto" if minutes == 1 else f"{minutes} minuti")
    if secs or not minutes:
        parts.append(f"{secs} secondo" if secs == 1 else f"{secs} secondi")
    return " e ".join(parts)


@dataclass
class _ActiveTimer:
    id: int
    label: str
    fire_at: float  # time.monotonic() di scadenza
    task: asyncio.Task | None = field(default=None, repr=False)


class TimerManager:
    """Gestisce i timer attivi e le notifiche di scadenza."""

    def __init__(self) -> None:
        self._timers: dict[int, _ActiveTimer] = {}
        self._counter = 0

    # ---------------------------------------------------------------- handlers
    def add(self, seconds: int, label: str) -> str:
        self._counter += 1
        tid = self._counter
        timer = _ActiveTimer(id=tid, label=label, fire_at=time.monotonic() + seconds)
        timer.task = asyncio.get_running_loop().create_task(self._fire(tid, seconds))
        self._timers[tid] = timer
        log.info("timer_set", id=tid, label=label, seconds=seconds)
        return f"Timer «{label}» impostato per {_format_duration(seconds)}."

    def list(self) -> str:
        if not self._timers:
            return "Non ci sono timer attivi."
        now = time.monotonic()
        righe = [
            f"«{t.label}»: {_format_duration(max(0, round(t.fire_at - now)))} rimanenti"
            for t in sorted(self._timers.values(), key=lambda t: t.fire_at)
        ]
        return "Timer attivi: " + "; ".join(righe) + "."

    def cancel(self, label: str) -> str:
        target = label.strip().lower()
        for t in list(self._timers.values()):
            if t.label.lower() == target:
                if t.task is not None:
                    t.task.cancel()
                self._timers.pop(t.id, None)
                log.info("timer_cancelled", id=t.id, label=t.label)
                return f"Timer «{t.label}» annullato."
        return f"Nessun timer chiamato «{label}»."

    # ----------------------------------------------------------------- interni
    async def _fire(self, tid: int, seconds: int) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        timer = self._timers.pop(tid, None)
        if timer is None:
            return
        log.info("timer_fired", id=tid, label=timer.label)
        await self._notify(timer.label)

    async def _notify(self, label: str) -> None:
        """Banner di notifica macOS + suono. Non blocca l'event loop."""
        title = "Jarvis — timer"
        message = f"È scaduto il timer «{label}»."
        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}" sound name "Glass"',
            )
            await proc.wait()
            sound = await asyncio.create_subprocess_exec("afplay", _ALERT_SOUND)
            await sound.wait()
        except FileNotFoundError:
            # Non su macOS o tool assente: la notifica è best-effort.
            log.warning("timer_notify_unavailable", label=label)


def register(registry, manager: TimerManager) -> None:  # noqa: ANN001
    """Registra i tre handler del timer sul registry, legati a ``manager``."""

    @registry.tool(
        name="imposta_timer",
        description=(
            "Imposta un timer che avvisa con una notifica alla scadenza. "
            "Usalo quando l'utente chiede di essere avvisato dopo un certo tempo."
        ),
    )
    def imposta_timer(
        durata_secondi: Annotated[
            int, Field(gt=0, description="Durata del timer in secondi")
        ],
        etichetta: Annotated[
            str, Field(description="Nome breve del timer, es. 'pasta' o 'pausa'")
        ] = "timer",
    ) -> str:
        return manager.add(durata_secondi, etichetta)

    @registry.tool(
        name="elenca_timer",
        description="Elenca i timer attualmente attivi e il tempo rimanente.",
    )
    def elenca_timer() -> str:
        return manager.list()

    @registry.tool(
        name="cancella_timer",
        description="Annulla un timer attivo, identificato dalla sua etichetta.",
    )
    def cancella_timer(
        etichetta: Annotated[str, Field(description="Etichetta del timer da annullare")],
    ) -> str:
        return manager.cancel(etichetta)
