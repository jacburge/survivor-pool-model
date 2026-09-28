"""Track your own entries' picks and survival across the season.

Analogous to rival_tracker.py, but for your own entries rather than
observed rivals. Not needed for Week 4 (all 10 entries start fresh, which
is exactly what run_weekly.py already assumes) -- becomes necessary from
Week 5 onward, once entries have picked different teams and some may have
already lost. used_teams_by_entry's output is what a per-entry-aware
allocation search needs in place of the current fresh-start simplification
(see portfolio.py; that search extension is a separate, not-yet-built
piece -- this module is just the record-keeping).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from survivor.data.storage import DEFAULT_STORE_ROOT
from survivor.data.team_keys import to_abbreviation

PICKS_FILE = DEFAULT_STORE_ROOT / "my_entries" / "picks.csv"
COLUMNS = ["entry_id", "week", "team", "survived"]


def _load(path: Path = PICKS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    picks = pd.read_csv(path)
    # CSV carries no dtype metadata: a "survived" column with no True/False
    # written yet (only pending NaNs) round-trips as float64, which then
    # rejects a real bool assignment in record_result with a casting error.
    picks["survived"] = picks["survived"].astype(object).where(picks["survived"].notna(), pd.NA)
    return picks


def record_pick(entry_id: str, week: int, team: str, path: Path = PICKS_FILE) -> None:
    """Record what an entry picked this week.

    Idempotent per (entry_id, week): rerunning replaces, not duplicates.
    survived starts unresolved (blank/NaN) until record_result is called --
    call this as soon as a pick is made, then record_result once the game
    is decided.
    """
    picks = _load(path)
    team = to_abbreviation(team)
    picks = picks[~((picks["entry_id"] == entry_id) & (picks["week"] == week))]
    new_row = pd.DataFrame([{"entry_id": entry_id, "week": week, "team": team, "survived": pd.NA}])
    updated = pd.concat([picks, new_row], ignore_index=True)
    # force object dtype: a column of all-NA otherwise infers float64, which
    # later rejects a real bool assignment in record_result with a casting error
    updated["survived"] = updated["survived"].astype(object)

    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)


def record_result(entry_id: str, week: int, survived: bool, path: Path = PICKS_FILE) -> None:
    """Resolve a previously recorded pick's outcome. Raises if no pick was recorded for it."""
    picks = _load(path)
    mask = (picks["entry_id"] == entry_id) & (picks["week"] == week)
    if not mask.any():
        raise ValueError(f"no recorded pick for entry {entry_id!r} week {week} -- call record_pick first")
    picks.loc[mask, "survived"] = survived
    picks.to_csv(path, index=False)


def used_teams(entry_id: str, path: Path = PICKS_FILE) -> set[str]:
    """Every team this entry has ever picked, regardless of outcome (a losing pick still used the team)."""
    picks = _load(path)
    return set(picks[picks["entry_id"] == entry_id]["team"])


def is_alive(entry_id: str, path: Path = PICKS_FILE) -> bool:
    """False only if some recorded pick for this entry resolved to a loss.

    An entry with no history, or only pending/won picks, counts as alive.
    """
    picks = _load(path)
    entry_picks = picks[picks["entry_id"] == entry_id]
    return not (entry_picks["survived"] == False).any()  # noqa: E712 (nullable column, `is False` misses it)


def alive_entries(entry_ids: list[str], path: Path = PICKS_FILE) -> list[str]:
    """Filter entry_ids down to the ones still alive."""
    return [entry_id for entry_id in entry_ids if is_alive(entry_id, path)]


def used_teams_by_entry(entry_ids: list[str], path: Path = PICKS_FILE) -> dict[str, set[str]]:
    """{entry_id: used teams} for each of entry_ids -- the per-entry history an allocation search needs."""
    return {entry_id: used_teams(entry_id, path) for entry_id in entry_ids}


def record_allocation(
    allocation: dict[str, int], week: int, entry_ids: list[str], path: Path = PICKS_FILE
) -> dict[str, str]:
    """Assign a team-count allocation (e.g. from greedy_local_allocation) to specific entry IDs.

    entry_ids must have exactly sum(allocation.values()) entries -- raises
    otherwise, since a silent mismatch would either drop an entry's pick or
    invent one. Which entry gets which team within the allocation is
    arbitrary (entries are only distinguishable by their own prior
    history, and a fresh allocation has none yet to prefer one assignment
    over another). Records each pick and returns entry_id -> team.
    """
    total = sum(allocation.values())
    if len(entry_ids) != total:
        raise ValueError(f"allocation has {total} entries but {len(entry_ids)} entry_ids were given")

    assignment = {}
    entry_iter = iter(entry_ids)
    for team, count in allocation.items():
        for _ in range(count):
            entry_id = next(entry_iter)
            assignment[entry_id] = team
            record_pick(entry_id, week, team, path=path)
    return assignment
