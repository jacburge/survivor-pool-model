"""What would the pipeline have recommended for Week 1, 2026, given no hindsight?

Pulls SurvivorGrid's /2026/1 page, which -- because we're viewing it well
after Week 1 happened -- also carries each team's actual result. This
script deliberately never reads that result column when making the
recommendation: only the win_probability, pick_percentage, and spread
columns are used, all of which are pre-game market data (confirmed: LAC's
79.8% week-1 win probability matches its -9.5 spread via the normal
approximation, so these are genuine week-1-specific values, not the site's
current/live numbers). The result is used once, at the very end, only to
report how the recommendation actually turned out -- never to make it.

Single entry, fresh start (no prior weeks to have used teams from, since
week 1 is the season's first week -- no simplification needed here, unlike
scoring a mid-season week).

Run: .venv/bin/python scripts/backtest_week1.py
"""

import numpy as np
import pandas as pd

from survivor.data.survivorgrid_client import (
    dedupe_schedule_games,
    fetch_week_html,
    parse_pick_grid,
    parse_schedule_grid,
)
from survivor.probability.current_week import DEFAULT_TIE_PROBABILITY
from survivor.probability.ratings import DEFAULT_RIDGE, fit_team_ratings
from survivor.simulation.field_simulator import score_candidate, simulate_rival_field

YEAR = 2026
WEEK = 1
FINAL_WEEK = 18
POT = 9000.0
N_RIVALS = 500  # assumed field size -- the real Week 1 field size for this specific pool is unknown
N_PATHS = 20000


def main() -> None:
    html = fetch_week_html(YEAR, WEEK)
    schedule_grid = parse_schedule_grid(html, start_week=WEEK)
    picks = parse_pick_grid(html)  # NOTE: 'result' column exists but is intentionally unused until the end

    all_games = dedupe_schedule_games(schedule_grid)
    full_schedule = all_games[["week", "home_team", "away_team"]]
    week1_games = all_games[all_games["week"] == WEEK][["home_team", "away_team", "home_spread"]]

    fit = fit_team_ratings(week1_games, ridge=DEFAULT_RIDGE)
    in_sample_mae = fit.residuals.abs().mean()
    print(f"Fitted ratings from {len(week1_games)} Week {WEEK} games "
          f"(in-sample MAE: {in_sample_mae:.3f} points, home-field advantage: {fit.home_field_advantage:.2f})")

    current_week_survival = {
        team: prob * (1 - DEFAULT_TIE_PROBABILITY)
        for team, prob in zip(picks["team"], picks["win_probability"])
        if pd.notna(prob)
    }

    print(f"\nRunning field simulation: {N_PATHS} paths, {N_RIVALS} rivals, Weeks {WEEK}-{FINAL_WEEK}...")
    rng = np.random.default_rng(2026)
    sim = simulate_rival_field(
        full_schedule, fit.ratings, fit.home_field_advantage,
        current_week=WEEK, final_week=FINAL_WEEK, n_paths=N_PATHS, n_rivals=N_RIVALS, pot=POT,
        current_week_survival_probability=current_week_survival, rng=rng,
    )

    week1_teams = sorted(week1_games["home_team"]) + sorted(week1_games["away_team"])
    results = []
    for team in week1_teams:
        payouts = score_candidate(sim, team)
        results.append(
            {
                "team": team,
                "expected_payout": payouts.mean(),
                "se": payouts.std(ddof=1) / np.sqrt(len(payouts)),
                "week1_survival_prob": current_week_survival.get(team, float("nan")),
            }
        )
    results_df = pd.DataFrame(results).sort_values("expected_payout", ascending=False).reset_index(drop=True)

    print("\nTop 10 by expected payout (the pipeline's actual objective):")
    print(results_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    by_survival = results_df.sort_values("week1_survival_prob", ascending=False)
    print("\nTop 10 by pure survival probability (the naive baseline the plan argues against):")
    print(by_survival.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    recommendation = results_df.iloc[0]["team"]
    print(f"\nPipeline recommendation for Week {WEEK}: {recommendation}")

    # Postscript only -- the actual result, never used above.
    actual_result = picks.set_index("team")["result"].get(recommendation)
    if actual_result is not None:
        outcome = "survived (won)" if actual_result == "W" else "would have been eliminated (lost)"
        print(f"(Postscript, not used in the recommendation: {recommendation} actually {outcome} in Week {WEEK}.)")


if __name__ == "__main__":
    main()
