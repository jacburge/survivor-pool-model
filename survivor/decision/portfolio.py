"""Joint allocation of your entries across candidate teams (Phase 6).

Ten entries all on one team is one bet at ten times the stake, so entries
must be optimized jointly rather than picking the best team for each entry
independently -- see the plan's Portfolio layer note. This module enumerates
every way to split n_entries across a short list of candidate teams (stars
and bars: choosing counts that sum to n_entries is equivalent to placing
len(candidate_teams) - 1 dividers among n_entries items) and leaves scoring
each allocation's expected payout to the field simulator.

MAX_ENTRIES: this season's actual Splash Sports league caps a single
participant at 25 entries, and that cap is enforced here as a hard input
validation rather than a soft warning, since silently truncating or scoring
past it would produce a recommendation you can't legally submit. This
number is specific to the current league, not a property of the model --
see the "League configuration (future)" note in plan.md's Phase 6 section
for how a multi-league version should handle this instead of hardcoding it.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from survivor.simulation.field_simulator import FieldSimulation, team_elimination_week

MAX_ENTRIES = 25  # this league's cap (Splash Sports, 2026 season) -- see module docstring


def validate_entry_count(n_entries: int) -> None:
    """Raises ValueError if n_entries is out of the range this league allows."""
    if n_entries < 1:
        raise ValueError(f"n_entries must be at least 1, got {n_entries}")
    if n_entries > MAX_ENTRIES:
        raise ValueError(
            f"n_entries ({n_entries}) exceeds this league's cap of {MAX_ENTRIES}. "
            "If you're running this for a different league, update MAX_ENTRIES in "
            "survivor/decision/portfolio.py -- see plan.md's Phase 6 design note."
        )


def enumerate_allocations(n_entries: int, candidate_teams: list[str]) -> list[dict[str, int]]:
    """Every way to split n_entries indistinguishable entries across candidate_teams.

    Returns one dict per allocation, mapping team -> entry count, omitting
    teams assigned zero entries. Count is C(n_entries + k - 1, k - 1) for k
    candidate teams -- 1,001 for the plan's "10 entries across the top 5"
    example, so exhaustive enumeration is cheap at this league's entry cap.
    A much larger cap or candidate list would need a non-exhaustive search
    instead of this stars-and-bars enumeration.
    """
    validate_entry_count(n_entries)
    if not candidate_teams:
        raise ValueError("candidate_teams must be non-empty")

    k = len(candidate_teams)
    allocations = []
    # stars and bars: choose k - 1 divider positions among n_entries + k - 1 slots
    for dividers in combinations(range(n_entries + k - 1), k - 1):
        bounds = (-1, *dividers, n_entries + k - 1)
        counts = [bounds[i + 1] - bounds[i] - 1 for i in range(k)]
        allocations.append({team: c for team, c in zip(candidate_teams, counts) if c > 0})
    return allocations


def score_allocation(
    sim: FieldSimulation,
    allocation: dict[str, int],
    elimination_weeks: dict[str, np.ndarray],
) -> np.ndarray:
    """Expected payout per path for a joint allocation of entries across teams.

    elimination_weeks must have one array (shape (sim.n_paths,), -1 where
    that team's entry survives the whole horizon) per team in `allocation`
    -- as returned by survivor.simulation.field_simulator.team_elimination_week.
    Compute each candidate team's array once and reuse it across every
    allocation that uses that team; it doesn't depend on how many entries
    you put there, only on the team and (for now) an empty used-teams
    history, so the same array is valid for every allocation this call.

    This is the fix field_simulator.score_candidate can't express: that
    function scores one entry as if it were your only one, splitting only
    against the rival field ("+1" for your entry). With several entries,
    two things it ignores start to matter: (1) several of your entries can
    survive to the final week together, splitting the pot between each
    other as well as with surviving rivals; (2) the week the *whole* field
    empties -- the event that triggers a cohort split -- has to account for
    your other still-alive entries, not just rivals. A rival-only "field
    emptied" week can be wrong once you have entries of your own still
    alive past it: an entry eliminated that week didn't actually die
    alongside the last survivors, because you had another entry keeping
    the field non-empty, so it isn't part of any split and should score
    zero, not a rival-cohort share it was never entitled to.
    """
    missing = set(allocation) - set(elimination_weeks)
    if missing:
        raise ValueError(f"elimination_weeks missing entries for: {sorted(missing)}")

    n_weeks = len(sim.weeks)
    n_paths = sim.n_paths

    # how many of your entries are still alive after each week, per path
    your_alive_after = np.zeros((n_weeks, n_paths), dtype=int)
    for team, count in allocation.items():
        elim = elimination_weeks[team]  # -1 = never eliminated within the horizon
        for w_idx, week in enumerate(sim.weeks):
            your_alive_after[w_idx] += count * ((elim == -1) | (elim > week))

    # rivals + yours, jointly -- this is the field the plan's payout rule
    # actually means, not rivals alone
    total_alive_after = sim.alive_count_by_week + your_alive_after

    is_zero = total_alive_after == 0
    has_emptied = is_zero.any(axis=0)
    first_zero_idx = is_zero.argmax(axis=0)  # 0 where has_emptied is False; unused there

    initial_total = sim.n_rivals + sum(allocation.values())
    before = np.vstack([np.full(n_paths, initial_total), total_alive_after[:-1]])
    cohort_size = before[first_zero_idx, np.arange(n_paths)]
    true_emptied_week = np.where(has_emptied, np.array(sim.weeks)[first_zero_idx], -1)

    my_full_survivors = np.zeros(n_paths, dtype=int)
    my_cohort = np.zeros(n_paths, dtype=int)
    for team, count in allocation.items():
        elim = elimination_weeks[team]
        my_full_survivors += count * (elim == -1)
        my_cohort += count * (elim == true_emptied_week)

    # guard both denominators against 0/0 on paths where the branch that
    # uses them isn't the one np.where ends up selecting (np.where still
    # evaluates both operands eagerly)
    survivor_denom = np.where(my_full_survivors > 0, sim.rival_survivors + my_full_survivors, 1)
    cohort_denom = np.where(my_cohort > 0, cohort_size, 1)

    return np.where(
        my_full_survivors > 0,
        sim.pot * my_full_survivors / survivor_denom,
        np.where(my_cohort > 0, sim.pot * my_cohort / cohort_denom, 0.0),
    )


def best_allocations(
    sim: FieldSimulation,
    candidate_teams: list[str],
    n_entries: int,
    used_teams_before: dict[str, set[str]] | None = None,
) -> list[tuple[dict[str, int], float, float]]:
    """Every allocation of n_entries across candidate_teams, ranked by mean expected payout.

    Returns (allocation, mean_payout, standard_error) tuples, best first.
    The standard error is of that allocation's payout alone, not of the gap
    to the runner-up -- per the plan's simulation-variance note, common
    random numbers (every allocation is scored against the same sim) make
    the gap's standard error much smaller than either allocation's own, so
    compare top candidates on the same run rather than reading these in
    isolation.
    """
    validate_entry_count(n_entries)
    used_teams_before = used_teams_before or {}

    elimination_weeks = {
        team: team_elimination_week(sim, team, used_teams_before.get(team))
        for team in candidate_teams
    }

    results = []
    for allocation in enumerate_allocations(n_entries, candidate_teams):
        payouts = score_allocation(sim, allocation, elimination_weeks)
        mean = float(payouts.mean())
        stderr = float(payouts.std(ddof=1) / np.sqrt(sim.n_paths)) if sim.n_paths > 1 else 0.0
        results.append((allocation, mean, stderr))

    results.sort(key=lambda result: result[1], reverse=True)
    return results
