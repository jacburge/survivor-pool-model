"""Validate Phase 3's acceptance criteria against cached/real data.

Checks: (1) fitted ratings reproduce real spreads with mean absolute error
under 1 point, and (2) projecting forward from that fit sits within ~1.5
points of the real Week 4 lookahead lines we already have cached.

Compares two fitting approaches:
- single-week: fit on Week 3's own real spreads alone (the original
  approach). No new API calls -- uses data_store/odds/latest.csv.
- season-to-date: fit on Weeks 1-3's real closing spreads combined (one
  SurvivorGrid pull per week, free/keyless, paced at 1 req/sec -- see
  fetch_season_to_date_games). More games per team should mean less
  single-week noise in the fit, which is most of what the lookahead error
  measures (see DEFAULT_SEASON_TO_DATE_RIDGE's comment in ratings.py for
  the full reasoning and the noise-vs-drift decomposition it's based on).

Run: .venv/bin/python scripts/validate_ratings.py
"""

import pandas as pd

from survivor.data.survivorgrid_client import fetch_season_to_date_games
from survivor.probability.current_week import compute_current_week_spreads
from survivor.probability.ratings import DEFAULT_RIDGE, DEFAULT_SEASON_TO_DATE_RIDGE, fit_team_ratings

IN_SAMPLE_MAE_BAR = 1.0
LOOKAHEAD_MAE_BAR = 1.5


def _connected_components(games: pd.DataFrame) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, game in games.iterrows():
        parent.setdefault(game["home_team"], game["home_team"])
        parent.setdefault(game["away_team"], game["away_team"])
        union(game["home_team"], game["away_team"])

    components: dict[str, set[str]] = {}
    for team in parent:
        components.setdefault(find(team), set()).add(team)
    return list(components.values())


def _lookahead_mae(fit, week4_spreads: pd.DataFrame) -> tuple[float, int]:
    known = set(fit.ratings)
    errors = []
    for _, game in week4_spreads.iterrows():
        if game["home_team"] not in known or game["away_team"] not in known:
            continue
        projected = -(fit.ratings[game["home_team"]] - fit.ratings[game["away_team"]] + fit.home_field_advantage)
        errors.append(abs(projected - game["home_spread"]))
    return sum(errors) / len(errors), len(errors)


def _report(label: str, fit, week4_spreads: pd.DataFrame) -> None:
    in_sample_mae = fit.residuals.abs().mean()
    lookahead_mae, n_games = _lookahead_mae(fit, week4_spreads)
    print(f"\n[{label}]")
    print(f"  In-sample MAE: {in_sample_mae:.3f} "
          f"({'PASS' if in_sample_mae < IN_SAMPLE_MAE_BAR else 'FAIL'}, bar {IN_SAMPLE_MAE_BAR})")
    print(f"  Week 4 lookahead MAE: {lookahead_mae:.3f} over {n_games} games "
          f"({'PASS' if lookahead_mae < LOOKAHEAD_MAE_BAR else 'FAIL'}, bar {LOOKAHEAD_MAE_BAR})")


def main() -> None:
    odds = pd.read_csv("data_store/odds/latest.csv")
    odds["commence_time"] = pd.to_datetime(odds["commence_time"])
    week3_odds = odds[odds["commence_time"] < "2026-10-01"]
    week4_odds = odds[odds["commence_time"] >= "2026-10-01"]
    week3_spreads = compute_current_week_spreads(week3_odds)
    week4_spreads = compute_current_week_spreads(week4_odds)

    components = _connected_components(week3_spreads)
    print(f"Week 3 alone schedule graph: {len(components)} connected component(s), "
          f"sizes {[len(c) for c in components]}")
    single_week_fit = fit_team_ratings(week3_spreads, ridge=DEFAULT_RIDGE)
    _report("single-week (Week 3 only)", single_week_fit, week4_spreads)

    print("\nPulling Weeks 1-3 real closing spreads from SurvivorGrid (free, ~3 requests)...")
    season_to_date = fetch_season_to_date_games(2026, through_week=3)
    components = _connected_components(season_to_date)
    print(f"Weeks 1-3 combined schedule graph: {len(components)} connected component(s), "
          f"sizes {[len(c) for c in components]}")

    season_fit_no_ridge = fit_team_ratings(season_to_date, ridge=0.0)
    _report("season-to-date, ridge=0.0", season_fit_no_ridge, week4_spreads)

    season_fit = fit_team_ratings(season_to_date, ridge=DEFAULT_SEASON_TO_DATE_RIDGE)
    _report(f"season-to-date, ridge={DEFAULT_SEASON_TO_DATE_RIDGE}", season_fit, week4_spreads)

    lookahead_mae, _ = _lookahead_mae(season_fit, week4_spreads)
    if lookahead_mae >= LOOKAHEAD_MAE_BAR:
        print(
            "\nStill not clearing the lookahead bar. Real improvement over the single-week fit "
            "(see ratings.py's DEFAULT_SEASON_TO_DATE_RIDGE comment for the full picture and what "
            "was tried), but not a full pass yet. Errors are broadly spread across games, not a "
            "couple of outliers, so the residual gap is plausibly real-time news between weeks a "
            "backward-looking spread fit can't see, plus still-limited history (3 games/team so "
            "far). Revisit as more real weeks accumulate -- rerun this script weekly."
        )


if __name__ == "__main__":
    main()
