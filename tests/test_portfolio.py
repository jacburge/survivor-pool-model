import math

import numpy as np
import pandas as pd
import pytest

from survivor.decision.portfolio import (
    MAX_ENTRIES,
    best_allocations,
    enumerate_allocations,
    score_allocation,
    validate_entry_count,
)
from survivor.simulation.field_simulator import (
    N_TEAMS,
    TEAM_INDEX,
    FieldSimulation,
    score_candidate,
    simulate_rival_field,
    team_elimination_week,
)


def test_validate_entry_count_accepts_the_league_cap():
    validate_entry_count(MAX_ENTRIES)  # no raise


def test_validate_entry_count_rejects_above_the_cap():
    with pytest.raises(ValueError):
        validate_entry_count(MAX_ENTRIES + 1)


def test_validate_entry_count_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        validate_entry_count(0)
    with pytest.raises(ValueError):
        validate_entry_count(-1)


def test_enumerate_allocations_count_matches_stars_and_bars():
    allocations = enumerate_allocations(10, ["DET", "LAC", "JAX", "KC", "BUF"])
    assert len(allocations) == math.comb(10 + 5 - 1, 5 - 1)


def test_enumerate_allocations_every_allocation_sums_to_n_entries():
    allocations = enumerate_allocations(10, ["DET", "LAC", "JAX"])
    for allocation in allocations:
        assert sum(allocation.values()) == 10


def test_enumerate_allocations_omits_zero_count_teams():
    allocations = enumerate_allocations(2, ["DET", "LAC", "JAX"])
    all_teams_used = {team for allocation in allocations for team in allocation}
    assert all_teams_used == {"DET", "LAC", "JAX"}
    assert all(0 not in allocation.values() for allocation in allocations)


def test_enumerate_allocations_single_team_gets_everything():
    allocations = enumerate_allocations(5, ["DET"])
    assert allocations == [{"DET": 5}]


def test_enumerate_allocations_rejects_above_the_cap():
    with pytest.raises(ValueError):
        enumerate_allocations(MAX_ENTRIES + 1, ["DET", "LAC"])


def test_enumerate_allocations_rejects_empty_candidate_list():
    with pytest.raises(ValueError):
        enumerate_allocations(10, [])


def _blank_field_simulation(weeks, n_paths, n_rivals, pot, alive_count_by_week=None):
    return FieldSimulation(
        weeks=weeks,
        n_paths=n_paths,
        n_rivals=n_rivals,
        pot=pot,
        true_ratings={},
        survival_probability=np.full((len(weeks), n_paths, N_TEAMS), 0.5),
        team_wins=np.zeros((len(weeks), n_paths, N_TEAMS), dtype=bool),
        playing=np.zeros((len(weeks), N_TEAMS), dtype=bool),
        rival_survivors=np.zeros(n_paths, dtype=int),
        field_emptied_week=np.full(n_paths, -1, dtype=int),
        emptying_cohort_size=np.zeros(n_paths, dtype=int),
        alive_count_by_week=(
            alive_count_by_week if alive_count_by_week is not None else np.zeros((len(weeks), n_paths), dtype=int)
        ),
    )


def test_score_allocation_single_entry_matches_full_horizon_survival_formula():
    # one entry, survives both weeks, 1 rival also survives to the end
    weeks = [4, 5]
    sim = _blank_field_simulation(
        weeks, n_paths=1, n_rivals=3, pot=900.0, alive_count_by_week=np.array([[2], [1]])
    )
    sim.rival_survivors = np.array([1])

    elimination_weeks = {"BUF": np.array([-1])}
    payouts = score_allocation(sim, {"BUF": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(900.0 / (1 + 1))  # matches score_candidate's pot/(rival_survivors + 1)


def test_score_allocation_two_surviving_entries_split_with_each_other_too():
    # two of your own entries both survive to the end, alongside 1 rival --
    # the fix: three-way split (900/3 = 300 each, 600 total), not two
    # independent 900/(1+1) = 450 scores summing to a nonsensical 900 total
    weeks = [4, 5]
    sim = _blank_field_simulation(
        weeks, n_paths=1, n_rivals=3, pot=900.0, alive_count_by_week=np.array([[2], [1]])
    )
    sim.rival_survivors = np.array([1])

    elimination_weeks = {"BUF": np.array([-1]), "KC": np.array([-1])}
    payouts = score_allocation(sim, {"BUF": 1, "KC": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(900.0 * 2 / 3)


def test_score_allocation_true_emptying_accounts_for_your_other_entries():
    # the core multi-entry bug: rivals all die at week 4, entry A dies at
    # week 4 alongside them, entry B (a different team) is still alive and
    # doesn't die until week 5. The rival-only "field emptied at week 4"
    # signal (what score_candidate alone would use) would wrongly pay A a
    # cohort share it isn't entitled to, since the real field -- including
    # your own still-alive entry B -- didn't actually empty until week 5.
    # Only B, which died in the week the WHOLE field (rivals + your other
    # entry) emptied, should be paid, and it should take the entire pot
    # since by week 5 no rivals and no other entries remain.
    weeks = [4, 5]
    sim = _blank_field_simulation(
        weeks, n_paths=1, n_rivals=2, pot=1000.0, alive_count_by_week=np.array([[0], [0]])
    )
    sim.rival_survivors = np.array([0])

    elimination_weeks = {"A": np.array([4]), "B": np.array([5])}
    payouts = score_allocation(sim, {"A": 1, "B": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(1000.0)  # all of it goes to B; A gets none of it


def test_score_allocation_no_survivors_and_no_matching_cohort_scores_zero():
    weeks = [4]
    sim = _blank_field_simulation(weeks, n_paths=1, n_rivals=5, pot=500.0, alive_count_by_week=np.array([[3]]))
    sim.rival_survivors = np.array([3])

    elimination_weeks = {"BUF": np.array([4])}  # eliminated week 4, rivals didn't empty that week
    payouts = score_allocation(sim, {"BUF": 1}, elimination_weeks)
    assert payouts[0] == pytest.approx(0.0)


def test_score_allocation_missing_team_in_elimination_weeks_raises():
    weeks = [4]
    sim = _blank_field_simulation(weeks, n_paths=1, n_rivals=5, pot=500.0)
    with pytest.raises(ValueError):
        score_allocation(sim, {"BUF": 1}, elimination_weeks={})


TWO_WEEK_SCHEDULE = pd.DataFrame(
    [
        {"week": 4, "home_team": "BUF", "away_team": "NYJ"},
        {"week": 4, "home_team": "KC", "away_team": "LV"},
        {"week": 5, "home_team": "BUF", "away_team": "MIA"},
        {"week": 5, "home_team": "KC", "away_team": "DEN"},
    ]
)


def test_score_allocation_matches_score_candidate_for_a_single_entry():
    # regression check on a real simulated sim (not hand-built): scoring
    # one entry via the portfolio path must agree with score_candidate
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(42)
    sim = simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=200, n_rivals=20, pot=1000.0, rng=rng,
    )
    expected = score_candidate(sim, "BUF")
    elimination_weeks = {"BUF": team_elimination_week(sim, "BUF")}
    actual = score_allocation(sim, {"BUF": 1}, elimination_weeks)
    np.testing.assert_allclose(actual, expected)


def test_best_allocations_sorted_best_first_and_allocations_sum_to_n_entries():
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(7)
    sim = simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=300, n_rivals=30, pot=1000.0, rng=rng,
    )
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3)

    means = [mean for _, mean, _ in results]
    assert means == sorted(means, reverse=True)
    assert all(sum(allocation.values()) == 3 for allocation, _, _ in results)
    assert len(results) == math.comb(3 + 2 - 1, 2 - 1)


def test_best_allocations_rejects_above_the_cap():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        best_allocations(sim, ["BUF", "KC"], n_entries=MAX_ENTRIES + 1)
