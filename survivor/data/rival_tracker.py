"""Record rival entries' revealed picks as they come in from Splash's picksheet.

No API -- Splash's picksheet visibility is manual (per the plan's Technical
dependencies: "Fall back to aggregate popularity model all season" if this
isn't available). This is just the storage mechanism; starting Week 4
Thursday, revealed picks get entered here, twice a week (after Thursday
kickoff and after the Sunday lock), per Phase 8.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from survivor.data.storage import DEFAULT_STORE_ROOT
from survivor.data.team_keys import to_abbreviation

PICKS_FILE = DEFAULT_STORE_ROOT / "rival_tracker" / "picks.csv"
COLUMNS = ["entry_id", "week", "team", "revealed_at"]


def _load(path: Path = PICKS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def record_pick(entry_id: str, week: int, team: str, revealed_at: str, path: Path = PICKS_FILE) -> None:
    """Append one revealed pick. Raises if this entry already has a pick recorded for this week."""
    picks = _load(path)
    team = to_abbreviation(team)

    existing = picks[(picks["entry_id"] == entry_id) & (picks["week"] == week)]
    if not existing.empty:
        raise ValueError(f"entry {entry_id!r} already has a recorded pick for week {week}: {existing.iloc[0]['team']}")

    new_row = pd.DataFrame([{"entry_id": entry_id, "week": week, "team": team, "revealed_at": revealed_at}])
    updated = pd.concat([picks, new_row], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)


def used_teams(entry_id: str, path: Path = PICKS_FILE) -> set[str]:
    """All teams a rival entry has used across every recorded week."""
    picks = _load(path)
    return set(picks[picks["entry_id"] == entry_id]["team"])


def pick_distribution(week: int, path: Path = PICKS_FILE) -> pd.Series:
    """Observed pick counts by team for a week, from entries revealed so far."""
    picks = _load(path)
    week_picks = picks[picks["week"] == week]
    return week_picks["team"].value_counts()
