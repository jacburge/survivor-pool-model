"""Persist each week's fitted team ratings, so the random-walk uncertainty
(ratings.DEFAULT_WEEKLY_RATING_STD) can eventually be calibrated from real
week-over-week changes in market ratings, per the plan's technical notes,
instead of the current placeholder tuned against elimination curves.

Storage is long format (one row per team per recorded week) so it can be
appended to incrementally without knowing all teams or weeks up front.
to_wide_by_season reshapes it into what ratings.fit_weekly_rating_std wants:
a list of wide (week x team) DataFrames, one per season, since diffs must
never cross a season boundary (see fit_weekly_rating_std's docstring).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from survivor.data.storage import DEFAULT_STORE_ROOT

RATING_HISTORY_PATH = DEFAULT_STORE_ROOT / "rating_history" / "history.csv"
COLUMNS = ["source", "year", "week", "team", "rating", "home_field_advantage"]


def _load(path: Path = RATING_HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def record_weekly_ratings(
    source: str,
    year: int,
    week: int,
    ratings: dict[str, float],
    home_field_advantage: float,
    path: Path = RATING_HISTORY_PATH,
) -> None:
    """Record one week's fitted ratings. Idempotent: rerunning the same
    (source, year, week) replaces its prior rows rather than duplicating them.
    """
    history = _load(path)
    history = history[~((history["source"] == source) & (history["year"] == year) & (history["week"] == week))]

    new_rows = pd.DataFrame(
        [
            {"source": source, "year": year, "week": week, "team": team,
             "rating": rating, "home_field_advantage": home_field_advantage}
            for team, rating in ratings.items()
        ]
    )
    updated = pd.concat([history, new_rows], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)


def load_rating_history(path: Path = RATING_HISTORY_PATH) -> pd.DataFrame:
    return _load(path)


def to_wide_by_season(history: pd.DataFrame) -> list[pd.DataFrame]:
    """One wide (week x team) DataFrame per (source, year), sorted by week.

    Ready to pass directly to ratings.fit_weekly_rating_std.
    """
    wide_frames = []
    for _, group in history.groupby(["source", "year"]):
        wide = group.pivot(index="week", columns="team", values="rating").sort_index()
        wide_frames.append(wide)
    return wide_frames
