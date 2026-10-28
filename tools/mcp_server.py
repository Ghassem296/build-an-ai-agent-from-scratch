
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Annotated, Any

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from tools.document_search import search_documents


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

mcp = FastMCP("web-search")

# امکان جایگزینی sleep در تست‌ها
_sleep = asyncio.sleep


def _search_sync(
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Run synchronous DDGS search in a worker thread."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "snippet": item.get("body", ""),
        }
        for item in results
    ]


@mcp.tool()
async def search_web(
    query: str,
    max_results: Annotated[int, Field(ge=1, le=10)] = 5,
) -> list[dict[str, Any]]:
    """Search public web sources and return titles, URLs, and snippets.

    Use this tool for:
    - Official documentation for public technologies and libraries.
    - Programming and technical questions requiring web research.
    - Current public information and source verification.

    For questions explicitly asking about official documentation,
    include the official project or documentation domain in the
    search query when known.

    Do not use this tool to search the application's local knowledge
    base or internal documents.

    Args:
        query: Focused web search query. For official documentation,
            include the project name and official domain when known.
        max_results: Maximum number of results, from 1 to 10.
    """
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")

    max_attempts = 3
    backoff = 1.0
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.to_thread(
                _search_sync,
                query,
                max_results,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Web search attempt %d/%d failed: %s",
                attempt,
                max_attempts,
                exc,
            )

            if attempt < max_attempts:
                await _sleep(backoff)
                backoff *= 2

    logger.error(
        "Web search failed after %d attempts: %s",
        max_attempts,
        last_error,
    )

    raise RuntimeError(
        "Web search is temporarily unavailable. Please try again later."
    ) from None


def _fetch_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fetch and decode a JSON API response."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    request_url = f"{url}?{urlencode(params)}"

    request = Request(
        request_url,
        headers={
            "User-Agent": "build-an-ai-agent-from-scratch/1.0"
        },
    )

    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _weather_description(code: int | None) -> str:
    """Convert a WMO weather code to a readable description."""
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    if code is None:
        return "Unknown"

    return descriptions.get(code, f"Weather code {code}")


@mcp.tool()
async def get_weather(
    city: Annotated[str, Field(min_length=1, max_length=100)],
) -> dict[str, Any]:
    """Get current weather and today's forecast for a city.

    Args:
        city: City name, for example Tehran or Berlin.
    """
    city = city.strip()

    if not city:
        raise ValueError("city cannot be empty")

    try:
        location_data = await asyncio.to_thread(
            _fetch_json,
            "https://geocoding-api.open-meteo.com/v1/search",
            {
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )

        locations = location_data.get("results", [])

        if not locations:
            return {
                "city": city,
                "error": f"No location found for '{city}'.",
            }

        place = locations[0]

        forecast = await asyncio.to_thread(
            _fetch_json,
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "apparent_temperature,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "hourly": (
                    "temperature_2m,precipitation_probability,weather_code"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,sunrise,sunset"
                ),
                "forecast_days": 1,
                "timezone": "auto",
            },
        )

    except Exception as exc:
        logger.warning(
            "Weather lookup failed (%s)",
            type(exc).__name__,
        )
        raise RuntimeError(
            "Weather data is temporarily unavailable. Please try again later."
        ) from None

    current = forecast.get("current", {})
    daily = forecast.get("daily", {})
    hourly = forecast.get("hourly", {})

    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    rain_chances = hourly.get("precipitation_probability", [])
    codes = hourly.get("weather_code", [])

    hourly_forecast = [
        {
            "time": times[i],
            "temperature_c": (
                temperatures[i] if i < len(temperatures) else None
            ),
            "precipitation_probability_percent": (
                rain_chances[i] if i < len(rain_chances) else None
            ),
            "conditions": _weather_description(
                codes[i] if i < len(codes) else None
            ),
        }
        for i in range(min(8, len(times)))
    ]

    def daily_value(key: str) -> Any:
        values = daily.get(key, [])
        return values[0] if values else None

    return {
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "city": place.get("name", city),
        "country": place.get("country"),
        "timezone": forecast.get("timezone"),
        "current": {
            "time": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation"),
            "conditions": _weather_description(
                current.get("weather_code")
            ),
        },
        "today": {
            "date": daily_value("time"),
            "conditions": _weather_description(
                daily_value("weather_code")
            ),
            "temperature_min_c": daily_value("temperature_2m_min"),
            "temperature_max_c": daily_value("temperature_2m_max"),
            "precipitation_probability_max_percent": daily_value(
                "precipitation_probability_max"
            ),
            "sunrise": daily_value("sunrise"),
            "sunset": daily_value("sunset"),
        },
        "hourly_forecast": hourly_forecast,
    }


@mcp.tool()
async def search_documents_tool(
    query: str,
    top_k: int = 3,
) -> list[dict[str, str | float]]:
    """Search the local knowledge base using semantic similarity.

    Args:
        query: The question or search query.
        top_k: Number of documents to return, from 1 to 10.
    """
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    return await asyncio.to_thread(
        search_documents,
        query,
        top_k,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
