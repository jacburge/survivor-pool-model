"""Scrape SurvivorGrid's weekly NFL survivor pick-percentage grid.

No official API or export -- survivorgrid.com/{year}/{week} is a plain,
server-rendered HTML table (confirmed via a raw pull, not a JS-only page),
so a direct requests + BeautifulSoup parse is reliable without a browser.

Two uses per the plan: Week 4's own national distribution feeds the
popularity model directly, and past seasons' grids fit the softmax
temperature (Phase 4).
"""

from __future__ import annotations

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from survivor.data.team_keys import to_abbreviation

BASE_URL = "https://www.survivorgrid.com"
USER_AGENT = "Mozilla/5.0 (compatible; survivor-pool-research/1.0)"
TEAM_PATTERN = re.compile(r"[A-Z]+")


def fetch_week_html(year: int, week: int) -> str:
    response = requests.get(f"{BASE_URL}/{year}/{week}", headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def parse_pick_grid(html: str) -> pd.DataFrame:
    """Parse the grid table into one row per team: expected_value, win_probability, pick_percentage, result.

    A team's cell reads e.g. "PHI" (upcoming week) or "PHI\xa0(W)" (a
    played week, with the result appended in a resultW/resultL span) -- the
    leading run of capital letters is the team abbreviation either way.
    result is "W", "L", or None (upcoming week, or a historical week with
    no game -- a bye).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="grid")
    if table is None:
        raise ValueError("could not find the pick grid table in the page")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        team_match = TEAM_PATTERN.match(cells[3].get_text())
        if team_match is None:
            continue
        result_span = cells[3].find("span", class_=["resultW", "resultL"])
        result = result_span["class"][0][-1] if result_span else None  # "resultW" -> "W", "resultL" -> "L"
        rows.append(
            {
                "team": to_abbreviation(team_match.group()),
                "expected_value": _parse_float(cells[0].get_text(strip=True)),
                "win_probability": _parse_percent(cells[1].get_text(strip=True)),
                "pick_percentage": _parse_percent(cells[2].get_text(strip=True)),
                "result": result,
            }
        )
    return pd.DataFrame(rows, columns=["team", "expected_value", "win_probability", "pick_percentage", "result"])


def parse_schedule_grid(html: str, start_week: int) -> pd.DataFrame:
    """Parse the same grid's per-week opponent/spread columns into one row per (team, week).

    The page for /{year}/{week} shows one "gc" column per remaining week,
    from start_week through 18, one row per team. A cell reads e.g.
    "ARI<br><span>-9.5</span>" (team is home, favored by 9.5) or
    "@BUF<br><span>+3</span>" (team is away, a 3-point underdog) or "BYE".
    spread is always the row team's own spread, matching
    ratings.fit_team_ratings' home_spread convention when is_home is True.

    Combined with parse_pick_grid's win_probability/pick_percentage for
    start_week specifically, this is enough to fit ratings and run the
    simulator from a single page pull -- no other odds source needed,
    including for weeks that have already passed this season (useful for
    backtesting what the pipeline would have recommended at the time).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="grid")
    if table is None:
        raise ValueError("could not find the pick grid table in the page")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        team_match = TEAM_PATTERN.match(cells[3].get_text())
        if team_match is None:
            continue
        team = to_abbreviation(team_match.group())

        for offset, cell in enumerate(cells[4:-1]):  # last cell is the "fv" star rating, not a week
            week = start_week + offset
            if "bye" in cell.get("class", []):
                rows.append({"team": team, "week": week, "opponent": None, "is_home": None,
                             "spread": None, "is_bye": True})
                continue
            is_home = "rd" not in cell.get("class", [])
            opponent_match = TEAM_PATTERN.search(cell.get_text())
            spread_span = cell.find("span", class_="spread")
            rows.append(
                {
                    "team": team,
                    "week": week,
                    "opponent": to_abbreviation(opponent_match.group()) if opponent_match else None,
                    "is_home": is_home,
                    "spread": _parse_float(spread_span.get_text(strip=True)) if spread_span else None,
                    "is_bye": False,
                }
            )
    return pd.DataFrame(rows, columns=["team", "week", "opponent", "is_home", "spread", "is_bye"])


def _parse_percent(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text.rstrip("%")) / 100.0
    except ValueError:
        return None  # blank cell placeholder -- seen as "-", "--", "N/A" across seasons


def _parse_float(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None  # blank cell placeholder -- seen as "-", "--", "N/A" across seasons


def fetch_pick_grid(year: int, week: int) -> pd.DataFrame:
    """One live pull for a single year/week, tagged with that year and week."""
    df = parse_pick_grid(fetch_week_html(year, week))
    df.insert(0, "week", week)
    df.insert(0, "year", year)
    return df


def fetch_historical_pick_grids(years: range, weeks: range, delay_seconds: float = 1.0) -> pd.DataFrame:
    """Pull multiple year/week grids for fitting the popularity model.

    delay_seconds throttles requests -- this is an unofficial, undocumented
    source with no published rate limit, so pace pulls to be a considerate
    scraper rather than to satisfy any stated quota.
    """
    frames = []
    for year in years:
        for week in weeks:
            try:
                frames.append(fetch_pick_grid(year, week))
            except requests.HTTPError:
                continue  # e.g. a week/year combination that doesn't exist
            time.sleep(delay_seconds)
    return pd.concat(frames, ignore_index=True)
