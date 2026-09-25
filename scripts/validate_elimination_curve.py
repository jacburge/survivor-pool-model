"""Validate the simulator's elimination curve against real historical survivor pools (Phase 5).

Done when: simulated elimination curves fall within historical survivor
pool ranges.

Real historical survival is reconstructed directly from SurvivorGrid's
published weekly pick_percentage (which sums to ~1.0 every week, confirming
it's already expressed as a share of the *remaining* field) and each
team's actual win/loss result: cumulative survival after week w = the
product, over weeks 1..w, of (1 - the losing teams' combined pick share
that week).

The comparison uses weeks *elapsed since a fresh start*, not absolute week
number: our simulator starts all rivals fresh at whatever current_week is
(a known simplification -- real per-rival history isn't available before
Phase 8's rival tracker), so its week 4 is dynamically like a real pool's
week 1, not a real pool's week 4 (which already reflects 3 real elimination
rounds). Comparing by elapsed week keeps this apples to apples.

Run: .venv/bin/python scripts/validate_elimination_curve.py
"""

import numpy as np
import pandas as pd

from survivor.probability.current_week import compute_current_week_spreads
from survivor.probability.ratings import DEFAULT_RIDGE, DEFAULT_WEEKLY_RATING_STD, fit_team_ratings
from survivor.simulation.field_simulator import simulate_rival_field

HISTORICAL_DATA_PATH = "data_store/survivorgrid_historical/pick_grids_with_results_2023_2025.csv"
CHECKPOINT_ELAPSED_WEEKS = [1, 5, 9, 12, 15]


def real_historical_curves(data: pd.DataFrame) -> dict[int, list[float]]:
    """{elapsed_week: [survival_fraction per season]} from real 2023-2025 data."""
    curves_by_year = {}
    for year in sorted(data["year"].unique()):
        survival = 1.0
        curve = {}
        for week in range(1, 19):
            week_rows = data[(data["year"] == year) & (data["week"] == week)].dropna(
                subset=["pick_percentage", "result"]
            )
            if not week_rows.empty:
                losing_share = week_rows.loc[week_rows["result"] == "L", "pick_percentage"].sum()
                survival *= 1 - losing_share
            curve[week] = survival
        curves_by_year[year] = curve

    return {
        elapsed: [curves_by_year[year][elapsed] for year in curves_by_year]
        for elapsed in CHECKPOINT_ELAPSED_WEEKS
    }


def main() -> None:
    historical = pd.read_csv(HISTORICAL_DATA_PATH)
    real_curves = real_historical_curves(historical)

    schedule = pd.read_csv("data_store/schedule/latest.csv")
    odds = pd.read_csv("data_store/odds/latest.csv")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"])
    week3_spreads = compute_current_week_spreads(odds[odds["commence_time"] < "2026-10-01"])
    fit = fit_team_ratings(week3_spreads, ridge=DEFAULT_RIDGE)

    rng = np.random.default_rng(7)
    sim = simulate_rival_field(
        schedule, fit.ratings, fit.home_field_advantage, weekly_rating_std=DEFAULT_WEEKLY_RATING_STD,
        current_week=4, final_week=18, n_paths=20000, n_rivals=500, pot=9000.0, rng=rng,
    )
    simulated_survival = sim.alive_count_by_week.mean(axis=1) / sim.n_rivals

    print(f"{'elapsed wk':>10} | {'simulated':>10} | {'real range':>20} | in range?")
    print("-" * 60)
    all_in_range = True
    for elapsed in CHECKPOINT_ELAPSED_WEEKS:
        sim_value = simulated_survival[elapsed - 1]  # week index 0 == current_week == elapsed week 1
        real_values = real_curves[elapsed]
        lo, hi = min(real_values), max(real_values)
        in_range = lo <= sim_value <= hi
        all_in_range &= in_range
        print(f"{elapsed:>10} | {sim_value:>10.4f} | [{lo:.4f}, {hi:.4f}]      | {'yes' if in_range else 'NO'}")

    print(f"\n{'All checkpoints within historical range' if all_in_range else 'NOT all checkpoints within historical range'}.")


if __name__ == "__main__":
    main()
