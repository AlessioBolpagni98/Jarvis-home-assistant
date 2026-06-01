"""Pacchetto dei tool e factory del registry di default (spec §7).

``build_default_registry`` assembla i tre tool inclusi (timer, meteo, web_search)
a partire dalla config. Aggiungere un tool nuovo significa scrivere una funzione
in un modulo qui e registrarla — l'orchestratore non cambia (FR6).
"""

from __future__ import annotations

from ..config import ToolsConfig
from . import timer as timer_mod
from . import weather as weather_mod
from . import web_search as web_search_mod
from .registry import Tool, ToolRegistry, ToolResult
from .timer import TimerManager

__all__ = ["Tool", "ToolRegistry", "ToolResult", "build_default_registry"]


def build_default_registry(cfg: ToolsConfig) -> ToolRegistry:
    """Crea il registry con timer, meteo e web_search configurati."""
    registry = ToolRegistry()
    # Il TimerManager è catturato dalle closure dei tool: resta vivo col registry.
    timer_mod.register(registry, TimerManager())
    weather_mod.register(registry, cfg.weather_default_location)
    web_search_mod.register(registry, cfg.brave_api_key, cfg.search_results)
    return registry
