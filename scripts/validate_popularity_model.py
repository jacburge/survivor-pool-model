"""Validate Phase 4's acceptance criterion against cached historical data (no new pulls).

Done when: on held-out historical weeks, the model's log loss beats both a
uniform baseline and a baseline proportional to win probability.

Trains (fits beta) on 2022-2024, holds out 2025 entirely -- a clean
season-level split, no leakage. gamma is fixed at 0, matching the plan's
own expectation that the public largely ignores future value.

Run: .venv/bin/python scripts/validate_popularity_model.py
"""

import pandas as pd

from survivor.decision.popularity import (
    cross_entropy,
    fit_beta_to_weeks,
    proportional_to_win_probability,
    weekly_cross_entropy,
)

TRAIN_YEARS = {2022, 2023, 2024}
HELD_OUT_YEARS = {2025}


def to_weeks(df: pd.DataFrame) -> list[dict]:
    weeks = []
    for (_, _), group in df.groupby(["year", "week"]):
        group = group.dropna(subset=["win_probability", "pick_percentage"])
        if len(group) < 2:
            continue
        weeks.append(
            {
                "win_probability": dict(zip(group["team"], group["win_probability"])),
                "pick_percentage": dict(zip(group["team"], group["pick_percentage"])),
            }
        )
    return weeks


def main() -> None:
    data = pd.read_csv("data_store/survivorgrid_historical/pick_grids_2022_2025.csv")

    train_weeks = to_weeks(data[data["year"].isin(TRAIN_YEARS)])
    held_out_weeks = to_weeks(data[data["year"].isin(HELD_OUT_YEARS)])
    print(f"Training on {len(train_weeks)} weeks ({sorted(TRAIN_YEARS)}), "
          f"holding out {len(held_out_weeks)} weeks ({sorted(HELD_OUT_YEARS)}).")

    fitted_beta = fit_beta_to_weeks(train_weeks)
    print(f"\nFitted beta: {fitted_beta:.3f} (gamma fixed at 0)")

    fitted_losses, uniform_losses, proportional_losses = [], [], []
    for week in held_out_weeks:
        win_prob, pick_pct = week["win_probability"], week["pick_percentage"]
        fitted_losses.append(weekly_cross_entropy(win_prob, pick_pct, beta=fitted_beta))
        uniform_losses.append(weekly_cross_entropy(win_prob, pick_pct, beta=0.0))
        total = sum(pick_pct.values())
        observed = {t: v / total for t, v in pick_pct.items()}
        proportional_losses.append(cross_entropy(proportional_to_win_probability(win_prob), observed))

    mean_fitted = sum(fitted_losses) / len(fitted_losses)
    mean_uniform = sum(uniform_losses) / len(uniform_losses)
    mean_proportional = sum(proportional_losses) / len(proportional_losses)

    print(f"\nHeld-out mean log loss ({len(held_out_weeks)} weeks):")
    print(f"  fitted softmax (beta={fitted_beta:.2f}): {mean_fitted:.4f}")
    print(f"  uniform baseline:                        {mean_uniform:.4f}")
    print(f"  proportional-to-win-probability baseline: {mean_proportional:.4f}")

    beats_uniform = mean_fitted < mean_uniform
    beats_proportional = mean_fitted < mean_proportional
    print(f"\nBeats uniform: {beats_uniform}. Beats proportional: {beats_proportional}.")
    if beats_uniform and beats_proportional:
        print("Phase 4 acceptance criterion met.")
    else:
        print("Phase 4 acceptance criterion NOT met.")


if __name__ == "__main__":
    main()
