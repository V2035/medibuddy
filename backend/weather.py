"""
weather.py

Thin wrapper around Open-Meteo's free geocoding + forecast APIs.

Design principle (per assignment non-negotiables):
    "The bot must never answer with a forecast it doesn't actually have."

So every function here either returns a clean, structured result with
real numbers from the API, or raises a WeatherLookupError with a clear
reason. There is NO fallback path that invents or estimates weather —
callers (the LangGraph nodes) are expected to catch WeatherLookupError
and route to an honest "I can't get weather for that" response, never
paper over it with a guess.
"""

from __future__ import annotations

import requests
from dataclasses import dataclass, field
from typing import Optional


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Fields we ask Open-Meteo for. Must be listed explicitly per the assignment
# note: fields not named here (like uv_index) simply won't appear in the response.
CURRENT_FIELDS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weather_code",
]

DAILY_FIELDS = [
    "precipitation_probability_max",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "uv_index_max",
]

REQUEST_TIMEOUT_SECONDS = 8


class WeatherLookupError(Exception):
    """
    Raised whenever we cannot honestly produce a real weather answer:
    location couldn't be resolved, the API errored, timed out, or returned
    something malformed. Callers must NOT catch this and substitute a guess;
    they must surface the failure to the user honestly.
    """
    def __init__(self, reason: str, stage: str):
        self.reason = reason
        self.stage = stage  # "geocoding" | "forecast"
        super().__init__(f"[{stage}] {reason}")


@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    admin1: Optional[str] = None  # state/region, useful when city names collide


@dataclass
class WeatherSnapshot:
    location: Location
    current: dict = field(default_factory=dict)
    daily: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)  # full raw API response, for traceability/debugging


def geocode_city(city_name: str) -> Location:
    """
    Resolve a free-text city name to coordinates using Open-Meteo's geocoding API.

    City names aren't unique (per assignment note re: "Springfield" / "Bhopal").
    We take the first result as a reasonable default. If geocoding returns
    zero results or errors out, we raise WeatherLookupError -- this is
    deliberately treated as the SAME class of failure as the weather API
    being down, per the assignment's instruction that both should route to
    the same honest fallback.
    """
    if not city_name or not city_name.strip():
        raise WeatherLookupError("No location provided.", stage="geocoding")

    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city_name.strip(), "count": 5},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise WeatherLookupError(f"Geocoding request failed: {e}", stage="geocoding") from e
    except ValueError as e:
        raise WeatherLookupError(f"Geocoding returned invalid JSON: {e}", stage="geocoding") from e

    results = data.get("results")
    if not results:
        raise WeatherLookupError(
            f"Could not resolve location '{city_name}' to any coordinates.",
            stage="geocoding",
        )

    first = results[0]
    return Location(
        name=first.get("name", city_name),
        latitude=first["latitude"],
        longitude=first["longitude"],
        country=first.get("country"),
        admin1=first.get("admin1"),
    )


def fetch_weather(location: Location) -> WeatherSnapshot:
    """
    Fetch live current + daily weather for a resolved Location.

    Raises WeatherLookupError on any failure -- network error, non-200,
    malformed response, or a response missing the 'current' block we
    explicitly asked for (which would indicate something silently went
    wrong upstream).
    """
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": ",".join(CURRENT_FIELDS),
        "daily": ",".join(DAILY_FIELDS),
        "timezone": "auto",
    }

    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise WeatherLookupError(f"Forecast request failed: {e}", stage="forecast") from e
    except ValueError as e:
        raise WeatherLookupError(f"Forecast returned invalid JSON: {e}", stage="forecast") from e

    if "current" not in data:
        raise WeatherLookupError(
            "Forecast response had no 'current' data block.", stage="forecast"
        )

    return WeatherSnapshot(
        location=location,
        current=data.get("current", {}),
        daily=data.get("daily", {}),
        raw=data,
    )


def get_weather_for_city(city_name: str) -> WeatherSnapshot:
    """
    Convenience entry point used by the LangGraph node: city name in,
    WeatherSnapshot out. Lets WeatherLookupError propagate unchanged --
    the graph node decides how to branch on it.
    """
    location = geocode_city(city_name)
    return fetch_weather(location)


if __name__ == "__main__":
    # Manual sanity check -- run this file directly to confirm the API
    # wiring works against the live Open-Meteo service:
    #     python weather.py "Bhopal"
    import sys
    import json

    city = sys.argv[1] if len(sys.argv) > 1 else "Bhopal"
    try:
        snap = get_weather_for_city(city)
        print(f"Resolved '{city}' -> {snap.location}")
        print("Current:", json.dumps(snap.current, indent=2))
        print("Daily:", json.dumps(snap.daily, indent=2))
    except WeatherLookupError as e:
        print(f"Honest failure ({e.stage}): {e.reason}")
