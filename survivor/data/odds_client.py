"""Client for The Odds API (https://the-odds-api.com), and parsing into a tidy frame.

Reads THE_ODDS_API_KEY from the environment (via a .env file, never hardcoded
or committed). One HTTP call per get_odds(); the caller decides how often to
poll -- the plan calls for a refresh right before each lock, not continuous
polling, to conserve API quota.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from survivor.data.team_keys import to_abbreviation

BASE_URL = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"


class OddsAPIError(RuntimeError):
    pass


class OddsAPIClient:
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get("THE_ODDS_API_KEY")
        if not api_key:
            raise OddsAPIError(
                "no API key provided and THE_ODDS_API_KEY not set in the environment"
            )
        self._api_key = api_key
        self.last_quota_remaining: int | None = None
        self.last_quota_used: int | None = None

    def get_odds(
        self,
        sport: str = NFL_SPORT_KEY,
        regions: str = "us",
        markets: str = "h2h,spreads",
        odds_format: str = "decimal",
    ) -> list[dict[str, Any]]:
        """Fetch current odds for a sport. Raises OddsAPIError on any non-2xx response."""
        response = requests.get(
            f"{BASE_URL}/sports/{sport}/odds",
            params={
                "apiKey": self._api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
            timeout=30,
        )
        if not response.ok:
            raise OddsAPIError(f"odds API request failed ({response.status_code}): {response.text}")

        self.last_quota_remaining = _int_or_none(response.headers.get("x-requests-remaining"))
        self.last_quota_used = _int_or_none(response.headers.get("x-requests-used"))
        return response.json()


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value is not None else None


def parse_odds_events(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw odds API events into a tidy long-format frame.

    One row per (game, bookmaker, market, outcome team). Team names are
    normalized to the shared abbreviation key. Spread rows carry a `point`
    column; moneyline (h2h) rows have point = NaN.
    """
    rows = []
    for event in events:
        game_id = event["id"]
        commence_time = event["commence_time"]
        home_team = to_abbreviation(event["home_team"])
        away_team = to_abbreviation(event["away_team"])
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "game_id": game_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_key,
                            "market": market_key,
                            "team": to_abbreviation(outcome["name"]),
                            "price": outcome["price"],
                            "point": outcome.get("point"),
                        }
                    )
    columns = [
        "game_id", "commence_time", "home_team", "away_team",
        "bookmaker", "market", "team", "price", "point",
    ]
    return pd.DataFrame(rows, columns=columns)
