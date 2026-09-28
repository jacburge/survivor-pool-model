"""Validate Phase 6's acceptance criteria against cached/real data (no new API calls).

Done when: the top allocation is the same across 5 random seeds and
survives a 2 percentage point shift in the leading team's win probability,
or the result is flagged as a near tie.

Also demonstrates the precompute-vs-recompute tradeoff behind
best_allocations' and greedy_local_allocation's `elimination_weeks`
parameter, with real timings: precomputing every team playing a week
(~32 teams) once, versus the two search methods' own runtime, since that's
what actually determines whether a weekly batch job should precompute
broadly (see portfolio.py's module docstring for the reasoning: it's cheap
here because it's linear in candidate teams, not combinatorial).

Run: .venv/bin/python scripts/validate_portfolio.py
"""

import time

import numpy as np
import pandas as pd

from survivor.decision.portfolio import best_allocations, greedy_local_allocation, score_allocation
from survivor.probability.current_week import compute_current_week_probabilities, compute_current_week_spreads
from survivor.probability.ratings import DEFAULT_RIDGE, fit_team_ratings
from survivor.simulation.field_simulator import simulate_rival_field, team_elimination_week

CURRENT_WEEK = 4
FINAL_WEEK = 18
POT = 9000.0
N_RIVALS = 500
N_PATHS = 5000
N_ENTRIES = 10
TOP_N_CANDIDATES = 5
PROBABILITY_SHIFT = 0.02  # 2 percentage points, per the plan's acceptance criterion


def load_inputs():
    schedule = pd.read_csv("data_store/schedule/latest.csv")
    odds = pd.read_csv("data_store/odds/latest.csv")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"])

    week3_spreads = compute_current_week_spreads(odds[odds["commence_time"] < "2026-10-01"])
    fit = fit_team_ratings(week3_spreads, ridge=DEFAULT_RIDGE)

    week4_odds = odds[odds["commence_time"] >= "2026-10-01"]
    week4_probs = compute_current_week_probabilities(week4_odds)
    survival = pd.concat(
        [
            week4_probs.set_index("home_team")["home_survival_probability"],
            week4_probs.set_index("away_team")["away_survival_probability"],
        ]
    )
    return schedule, fit, survival.to_dict()


def top_allocation_identical(results_by_seed: list[dict]) -> bool:
    first = results_by_seed[0]
    return all(r == first for r in results_by_seed[1:])


def main() -> None:
    schedule, fit, week4_survival = load_inputs()
    candidates = sorted(week4_survival, key=week4_survival.get, reverse=True)[:TOP_N_CANDIDATES]
    leading_team = candidates[0]
    print(f"Top {TOP_N_CANDIDATES} Week {CURRENT_WEEK} candidates by survival probability: {candidates}")
    print(f"Leading team: {leading_team} ({week4_survival[leading_team]:.3f})")

    # --- Criterion 1: same top allocation across 5 random seeds ---
    top_allocations = []
    first_seed_results = None
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sim = simulate_rival_field(
            schedule, fit.ratings, fit.home_field_advantage,
            current_week=CURRENT_WEEK, final_week=FINAL_WEEK, n_paths=N_PATHS, n_rivals=N_RIVALS, pot=POT,
            current_week_survival_probability=week4_survival, rng=rng,
        )
        results = best_allocations(sim, candidates, n_entries=N_ENTRIES)
        if seed == 0:
            first_seed_results = results
        top_allocations.append(results[0].allocation)
        print(f"  seed {seed}: top allocation {results[0].allocation}, mean payout {results[0].mean_payout:.2f}")

    seeds_stable = top_allocation_identical(top_allocations)
    print(f"\nSame top allocation across 5 seeds: {seeds_stable}")

    if not seeds_stable:
        # Cross-seed instability alone can't tell you *why* -- it conflates
        # "these allocations are genuinely close" with "Monte Carlo noise
        # moved the apparent winner around." best_allocations already
        # computed a much more direct answer for free: gap_to_best is a
        # *paired* comparison (common random numbers), so its standard
        # error is usually far smaller than either allocation's own raw SE
        # -- exactly what's needed to tell a real near-tie from a precision
        # problem, without spending any more simulation budget.
        print("\nNot stable across seeds -- checking whether the top allocations are a genuine near-tie "
              "(paired gap_to_best from seed 0, reusing what best_allocations already computed):")
        tied_count = 0
        for r in first_seed_results[:10]:
            tied = r.gap_to_best <= 2 * r.gap_to_best_se
            tied_count += tied
            print(f"    {r.allocation}  mean={r.mean_payout:7.2f}  gap_to_best={r.gap_to_best:5.2f} "
                  f"+/- {r.gap_to_best_se:.2f}  ({'tied' if tied else 'distinguishable'})")
        if tied_count >= 5:
            print(f"\n{tied_count} of the top 10 allocations are statistically tied with the best one "
                  "(paired gap within 2 SE) -- this is a genuine near-tie, not insufficient precision. "
                  "More simulation paths would narrow the confidence interval around a near-zero true "
                  "gap, not produce a different winner.")
            seeds_stable = "near_tie"

    # --- Criterion 2: survives a 2pp shift in the leading team's probability ---
    shifted_probs = dict(week4_survival)
    shifted_probs[leading_team] = min(shifted_probs[leading_team] + PROBABILITY_SHIFT, 0.999)

    sim_baseline = simulate_rival_field(
        schedule, fit.ratings, fit.home_field_advantage,
        current_week=CURRENT_WEEK, final_week=FINAL_WEEK, n_paths=N_PATHS, n_rivals=N_RIVALS, pot=POT,
        current_week_survival_probability=week4_survival, rng=np.random.default_rng(0),
    )
    baseline_top = best_allocations(sim_baseline, candidates, n_entries=N_ENTRIES)[0].allocation

    sim_shifted = simulate_rival_field(
        schedule, fit.ratings, fit.home_field_advantage,
        current_week=CURRENT_WEEK, final_week=FINAL_WEEK, n_paths=N_PATHS, n_rivals=N_RIVALS, pot=POT,
        current_week_survival_probability=shifted_probs, rng=np.random.default_rng(0),
    )
    shifted_top = best_allocations(sim_shifted, candidates, n_entries=N_ENTRIES)[0].allocation

    print(f"\nBaseline top allocation:               {baseline_top}")
    print(f"Top allocation after +{PROBABILITY_SHIFT*100:.0f}pp on {leading_team}: {shifted_top}")
    probability_shift_stable = baseline_top == shifted_top
    print(f"Survives the probability shift: {probability_shift_stable}")

    if not probability_shift_stable:
        print("\nDoes not meet Phase 6's acceptance criterion: fails the probability-shift check.")
    elif seeds_stable is True:
        print("\nPhase 6 acceptance criterion met: clean pass.")
    elif seeds_stable == "near_tie":
        print("\nPhase 6 acceptance criterion met: confirmed near-tie, which the plan's own "
              "'or the result is flagged as a near tie' clause explicitly allows.")
    else:
        print("\nNot stable across seeds, and the top allocations were NOT confirmed as a near-tie "
              "(paired gap_to_best check above) -- this is a genuine, unresolved instability, not just "
              "Monte Carlo noise. Worth investigating further before trusting the recommendation.")

    # --- Precompute-vs-recompute timing, answering the "precompute all teams" question ---
    print("\n--- Precompute timing ---")
    all_teams_playing = sorted(set(schedule[schedule["week"] == CURRENT_WEEK]["home_team"])
                                | set(schedule[schedule["week"] == CURRENT_WEEK]["away_team"]))
    print(f"Teams playing Week {CURRENT_WEEK}: {len(all_teams_playing)}")

    start = time.time()
    all_elimination_weeks = {team: team_elimination_week(sim_baseline, team) for team in all_teams_playing}
    precompute_elapsed = time.time() - start
    print(f"Precomputing elimination_weeks for all {len(all_teams_playing)} teams: {precompute_elapsed:.2f}s")

    start = time.time()
    exhaustive_result = best_allocations(sim_baseline, candidates, n_entries=N_ENTRIES)[0]
    exhaustive_elapsed = time.time() - start
    print(f"Exhaustive search, top {TOP_N_CANDIDATES} candidates: {exhaustive_elapsed:.2f}s "
          f"(mean payout {exhaustive_result.mean_payout:.2f}, SE {exhaustive_result.standard_error:.2f})")

    start = time.time()
    greedy_result = greedy_local_allocation(
        sim_baseline, all_teams_playing, n_entries=N_ENTRIES, elimination_weeks=all_elimination_weeks
    )
    greedy_elapsed = time.time() - start
    print(f"Greedy+local search, all {len(all_teams_playing)} candidates (reusing precomputed weeks): "
          f"{greedy_elapsed:.2f}s (mean payout {greedy_result.mean_payout:.2f}, SE {greedy_result.standard_error:.2f})")

    diff = score_allocation(sim_baseline, greedy_result.allocation, all_elimination_weeks) - score_allocation(
        sim_baseline, exhaustive_result.allocation, all_elimination_weeks
    )
    gap, gap_se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(sim_baseline.n_paths))
    print(f"Greedy vs. exhaustive gap (paired): {gap:.2f}, SE {gap_se:.2f} "
          f"({'a real difference' if abs(gap) > 2 * gap_se else 'within noise at this path count'})")


if __name__ == "__main__":
    main()
