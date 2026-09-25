"""Validate Phase 2's acceptance criteria against cached data (no new API calls).

Checks: (1) each game's two survival probabilities plus tie probability
sum to 1, and (2) our consensus win probabilities sit within ~1.5
percentage points of a published devigged consensus -- here, SurvivorGrid's
own W% column, which they describe as "based on consensus moneyline from
the betting market," for the same week's games.

Runs both devig methods and compares, since the plan calls out Shin vs.
power as worth comparing on heavy favorites specifically. This is the
evidence behind current_week.py's DEFAULT_DEVIG_METHOD choice.

Run: .venv/bin/python scripts/validate_current_week.py
"""

import pandas as pd

from survivor.probability.current_week import compute_current_week_probabilities

ACCEPTABLE_ERROR_PCT_POINTS = 1.5


def compare_method(week_odds: pd.DataFrame, survivorgrid: pd.Series, method: str) -> pd.DataFrame:
    current_week = compute_current_week_probabilities(week_odds, method=method)

    totals = (
        current_week["home_survival_probability"]
        + current_week["away_survival_probability"]
        + current_week["tie_probability"]
    )
    assert (totals.round(9) == 1.0).all(), f"[{method}] found rows where probabilities don't sum to 1"

    our_win_prob = current_week.set_index("home_team")["home_win_probability"]
    our_win_prob = pd.concat(
        [our_win_prob, current_week.set_index("away_team")["away_win_probability"]]
    ).rename(f"{method}_win_probability")

    comparison = pd.concat([our_win_prob, survivorgrid], axis=1).dropna()
    comparison[f"{method}_abs_diff_pct_points"] = (
        (comparison[f"{method}_win_probability"] - comparison["survivorgrid_win_probability"]).abs() * 100
    )
    return comparison


def main() -> None:
    odds = pd.read_csv("data_store/odds/latest.csv")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"])

    # The cached odds pull spans the tail of Week 3 and all of Week 4; isolate
    # Week 3 here since that's the week we also have cached SurvivorGrid data for.
    week3_odds = odds[odds["commence_time"] < "2026-10-01"]

    survivorgrid = pd.read_csv("data_store/survivorgrid/latest.csv")
    survivorgrid = survivorgrid.set_index("team")["win_probability"].rename("survivorgrid_win_probability")

    shin = compare_method(week3_odds, survivorgrid, "shin")
    power = compare_method(week3_odds, survivorgrid, "power")
    print(f"Sum-to-1 check: passed for all games under both methods.\n")

    combined = pd.concat(
        [survivorgrid, shin["shin_abs_diff_pct_points"], power["power_abs_diff_pct_points"]], axis=1
    ).dropna()
    print(f"Compared {len(combined)} teams against SurvivorGrid's published consensus:")
    print(combined.sort_values("power_abs_diff_pct_points", ascending=False).to_string(
        float_format=lambda x: f"{x:.4f}"
    ))

    for method, df in [("shin", shin), ("power", power)]:
        col = f"{method}_abs_diff_pct_points"
        print(f"\n{method}: mean abs diff {df[col].mean():.2f} pct points, max {df[col].max():.2f} pct points")
        failing = df[df[col] > ACCEPTABLE_ERROR_PCT_POINTS]
        if failing.empty:
            print(f"  All teams within {ACCEPTABLE_ERROR_PCT_POINTS} pct points. Acceptance criterion met.")
        else:
            print(f"  {len(failing)} team(s) exceed {ACCEPTABLE_ERROR_PCT_POINTS} pct points.")


if __name__ == "__main__":
    main()
