"""Tool ``meteo`` — Open-Meteo (spec §7.2).

Open-Meteo è gratuito e **senza API key**: si inviano solo le coordinate (nessun
dato personale). La località si risolve dal nome con la geocoding API; se l'utente
non la specifica, si usa quella di default da config (``[tools].weather_default_location``).
"""

from __future__ import annotations

from typing import Annotated

import httpx
from pydantic import Field

from ..logging_setup import get_logger

log = get_logger(__name__)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 10.0

# Codici meteo WMO → descrizione italiana (sottoinsieme sufficiente).
_WMO: dict[int, str] = {
    0: "sereno",
    1: "prevalentemente sereno",
    2: "parzialmente nuvoloso",
    3: "coperto",
    45: "nebbia",
    48: "nebbia con brina",
    51: "pioviggine leggera",
    53: "pioviggine moderata",
    55: "pioviggine intensa",
    61: "pioggia debole",
    63: "pioggia moderata",
    65: "pioggia forte",
    71: "neve debole",
    73: "neve moderata",
    75: "neve abbondante",
    80: "rovesci deboli",
    81: "rovesci moderati",
    82: "rovesci violenti",
    95: "temporale",
    96: "temporale con grandine",
    99: "temporale con grandine forte",
}


def register(registry, default_location: str) -> None:  # noqa: ANN001
    """Registra il tool ``meteo`` sul registry."""

    @registry.tool(
        name="meteo",
        description=(
            "Fornisce le condizioni meteo attuali (temperatura, cielo, vento) di una "
            "località. Se l'utente non indica la città, usa quella predefinita."
        ),
    )
    async def meteo(
        localita: Annotated[
            str,
            Field(description="Nome della città, es. 'Roma'. Vuoto = località predefinita."),
        ] = "",
    ) -> str:
        place = localita.strip() or default_location
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            geo = await client.get(
                _GEOCODE_URL,
                params={"name": place, "count": 1, "language": "it", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                return f"Non trovo la località «{place}»."
            loc = results[0]
            lat, lon = loc["latitude"], loc["longitude"]
            nome = loc.get("name", place)

            fc = await client.get(
                _FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
            )
            fc.raise_for_status()
            cur = fc.json()["current"]

        cielo = _WMO.get(int(cur["weather_code"]), "condizioni variabili")
        temp = round(cur["temperature_2m"])
        perc = round(cur["apparent_temperature"])
        vento = round(cur["wind_speed_10m"])
        log.info("weather_ok", place=nome, temp=temp, code=cur["weather_code"])
        return (
            f"A {nome} ci sono {temp} gradi (percepiti {perc}), {cielo}, "
            f"vento a {vento} km/h."
        )
