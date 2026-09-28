import math

import numpy as np
import pandas as pd
import pytest

from survivor.decision.portfolio import (
    MAX_ENTRIES,
    best_allocations,
    enumerate_allocations,
    greedy_local_allocation,
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

    means = [result.mean_payout for result in results]
    assert means == sorted(means, reverse=True)
    assert all(sum(result.allocation.values()) == 3 for result in results)
    assert len(results) == math.comb(3 + 2 - 1, 2 - 1)


def test_best_allocations_rejects_above_the_cap():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        best_allocations(sim, ["BUF", "KC"], n_entries=MAX_ENTRIES + 1)


def _real_sim(n_paths=300, n_rivals=30, seed=7):
    base_ratings = {team: 0.0 for team in ["BUF", "NYJ", "KC", "LV", "MIA", "DEN"]}
    rng = np.random.default_rng(seed)
    return simulate_rival_field(
        TWO_WEEK_SCHEDULE, base_ratings, home_field_advantage=1.0, weekly_rating_std=1.0,
        current_week=4, final_week=5, n_paths=n_paths, n_rivals=n_rivals, pot=1000.0, rng=rng,
    )


def test_best_allocations_gap_to_best_is_zero_for_the_top_result():
    sim = _real_sim()
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3)
    assert results[0].gap_to_best == pytest.approx(0.0)
    assert results[0].gap_to_best_se == pytest.approx(0.0)


def test_best_allocations_gap_to_best_matches_mean_difference_for_others():
    sim = _real_sim()
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3)
    for result in results[1:]:
        assert result.gap_to_best == pytest.approx(results[0].mean_payout - result.mean_payout, abs=1e-6)
        assert result.gap_to_best_se >= 0.0


def test_best_allocations_accepts_precomputed_elimination_weeks():
    sim = _real_sim()
    precomputed = {team: team_elimination_week(sim, team) for team in ["BUF", "KC"]}
    results = best_allocations(sim, ["BUF", "KC"], n_entries=3, elimination_weeks=precomputed)
    fresh = best_allocations(sim, ["BUF", "KC"], n_entries=3)
    assert [r.mean_payout for r in results] == [r.mean_payout for r in fresh]


def test_best_allocations_precomputed_elimination_weeks_missing_team_raises():
    sim = _real_sim()
    with pytest.raises(ValueError):
        best_allocations(sim, ["BUF", "KC"], n_entries=3, elimination_weeks={"BUF": team_elimination_week(sim, "BUF")})


def test_greedy_local_allocation_matches_exhaustive_search_on_a_small_case():
    # the real validation: on a candidate set small enough to enumerate
    # exactly, the heuristic should land on (or statistically tie) the true
    # best allocation, not just something plausible-looking.
    sim = _real_sim(n_paths=2000, seed=11)
    exhaustive = best_allocations(sim, ["BUF", "KC", "NYJ"], n_entries=5)
    greedy_result = greedy_local_allocation(sim, ["BUF", "KC", "NYJ"], n_entries=5)

    assert greedy_result.allocation == exhaustive[0].allocation


def test_greedy_local_allocation_rejects_above_the_cap():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        greedy_local_allocation(sim, ["BUF", "KC"], n_entries=MAX_ENTRIES + 1)


def test_greedy_local_allocation_rejects_empty_candidate_list():
    sim = _blank_field_simulation([4], n_paths=1, n_rivals=1, pot=100.0)
    with pytest.raises(ValueError):
        greedy_local_allocation(sim, [], n_entries=3)


def test_greedy_local_allocation_accepts_precomputed_elimination_weeks():
    sim = _real_sim()
    precomputed = {team: team_elimination_week(sim, team) for team in ["BUF", "KC"]}
    result = greedy_local_allocation(sim, ["BUF", "KC"], n_entries=3, elimination_weeks=precomputed)
    assert sum(result.allocation.values()) == 3


def test_greedy_local_allocation_scales_to_a_large_candidate_list():
    # the actual point of this function: candidate lists too large for
    # best_allocations to enumerate exhaustively (all 32 teams would be
    # C(24+32-1, 31) allocations for 24 entries -- intractable).
    sim = _real_sim(n_paths=50, n_rivals=20)
    result = greedy_local_allocation(sim, list(TEAM_INDEX), n_entries=10)
    assert sum(result.allocation.values()) == 10
    assert set(result.allocation) <= set(TEAM_INDEX)
