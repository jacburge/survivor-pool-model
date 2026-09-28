"""Weekly production run (Phase 7): refresh real data, recommend an allocation
of your entries across this week's candidate teams, and archive the result.

Pulls a fresh schedule (this week through Week 18, free/ESPN) and live
current-week odds (The Odds API, uses quota). The odds pull is ON by
default here, unlike refresh_all.py's dev-focused default -- this script's
entire purpose is a real weekly decision, exactly the case the odds-API
conservation guidance carves out as worth spending quota on. Pass
--skip-refresh to reuse whatever is already in data_store/ instead (e.g.
if you already refreshed today and just want to re-run with different
--n-entries or --n-paths).

Fits current-week ratings from real spreads (Phase 3), gets real
current-week survival probabilities via Shin devig (Phase 2), runs the
field simulator through Week 18 (Phase 5), then recommends an allocation
via greedy_local_allocation over every team playing this week (Phase 6) --
found to meaningfully beat exhaustive search restricted to a handful of
top candidates (plan.md's Phase 6 section has the real numbers: a paired
gap of 47.5 +/- 9.96 on real Week 4, 2026 data). Also reports the
exhaustive top-5-by-survival-probability result as a sanity cross-check,
using the same simulated paths so the comparison is paired, not independent.

Known simplification carried over from the simulator: your entries and the
rival field both start fresh (no prior used teams) at --week. Correct if
none of your entries have picks locked in before that week; if they do,
this needs a per-entry used-team tracker first (analogous to
survivor.data.rival_tracker, but for your own entries) -- not built yet.

Runtime scales with --n-paths: precomputing elimination arrays for every
team playing (needed for the all-candidate greedy search) is the added
cost beyond Phase 5's own validated field-simulation runtime, since it's
one Hungarian-assignment solve per team, not just per candidate. Rough
budget at 500 rivals, ~32 teams playing: 20,000 paths ~9 minutes, 80,000
paths ~35-40 minutes (over Phase 5's 30-minute bar) -- use a lower
--n-paths for a quick look and raise it for the final pre-lock run.

Run: .venv/bin/python scripts/run_weekly.py --week 4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from survivor.data import schedule_client
from survivor.data.odds_client import OddsAPIClient, parse_odds_events
from survivor.data.storage import DEFAULT_STORE_ROOT, save_raw_pull
from survivor.decision.portfolio import best_allocations, greedy_local_allocation, score_allocation
from survivor.probability.current_week import compute_current_week_probabilities, compute_current_week_spreads
from survivor.probability.ratings import DEFAULT_RIDGE, fit_team_ratings
from survivor.simulation.field_simulator import simulate_rival_field, team_elimination_week

FINAL_WEEK = 18
TOP_N_FOR_SANITY_CHECK = 5


def refresh_schedule(year: int, week: int) -> pd.DataFrame:
    weeks = range(week, FINAL_WEEK + 1)
    raw_by_week = {w: schedule_client.fetch_week_schedule(w, year) for w in weeks}
    save_raw_pull("schedule", raw_by_week)
    parsed = pd.concat(
        [schedule_client.parse_week_schedule(raw, w) for w, raw in raw_by_week.items()], ignore_index=True
    )
    return schedule_client.add_effective_lock_times(parsed)


def refresh_odds() -> pd.DataFrame:
    client = OddsAPIClient()
    events = client.get_odds()
    save_raw_pull("the_odds_api_nfl", events)
    odds = parse_odds_events(events)
    print(f"  {odds['game_id'].nunique()} games. Quota -- used: {client.last_quota_used}, "
          f"remaining: {client.last_quota_remaining}")
    return odds


def load_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = pd.read_csv(DEFAULT_STORE_ROOT / "schedule" / "latest.csv")
    odds = pd.read_csv(DEFAULT_STORE_ROOT / "odds" / "latest.csv")
    return schedule, odds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, required=True, help="the current week to decide for")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--n-entries", type=int, default=10)
    parser.add_argument("--n-rivals", type=int, default=500, help="assumed field size -- unknown until lock day")
    parser.add_argument("--n-paths", type=int, default=20000, help="see docstring for the runtime-vs-precision tradeoff")
    parser.add_argument("--pot", type=float, default=9000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-refresh", action="store_true", help="reuse data_store/ as-is (saves an odds API call)")
    args = parser.parse_args()

    if args.skip_refresh:
        print("Skipping refresh, using cached data_store/ as-is.")
        schedule, odds = load_cached()
    else:
        print(f"Refreshing schedule (weeks {args.week}-{FINAL_WEEK}, {args.year})...")
        schedule = refresh_schedule(args.year, args.week)
        print("Refreshing current-week odds (live API call)...")
        odds = refresh_odds()

    odds["commence_time"] = pd.to_datetime(odds["commence_time"])
    week_games = schedule[schedule["week"] == args.week]
    valid_pairs = set(zip(week_games["home_team"], week_games["away_team"]))
    week_odds = odds[[(h, a) in valid_pairs for h, a in zip(odds["home_team"], odds["away_team"])]]
    if week_odds.empty:
        raise SystemExit(
            f"No odds rows matched Week {args.week}'s schedule -- odds pull may not cover this "
            "week yet (lookahead is only ~1.5 weeks), or --week doesn't match what's live."
        )

    week_spreads = compute_current_week_spreads(week_odds)
    fit = fit_team_ratings(week_spreads, ridge=DEFAULT_RIDGE)
    print(f"\nFitted ratings from {len(week_spreads)} Week {args.week} games "
          f"(home-field advantage: {fit.home_field_advantage:.2f})")

    week_probs = compute_current_week_probabilities(week_odds)
    current_week_survival = pd.concat(
        [
            week_probs.set_index("home_team")["home_survival_probability"],
            week_probs.set_index("away_team")["away_survival_probability"],
        ]
    ).to_dict()

    print(f"\nRunning field simulation: {args.n_paths} paths, {args.n_rivals} rivals, "
          f"Weeks {args.week}-{FINAL_WEEK}...")
    rng = np.random.default_rng(args.seed)
    sim = simulate_rival_field(
        schedule, fit.ratings, fit.home_field_advantage,
        current_week=args.week, final_week=FINAL_WEEK, n_paths=args.n_paths, n_rivals=args.n_rivals, pot=args.pot,
        current_week_survival_probability=current_week_survival, rng=rng,
    )

    all_teams_playing = sorted(set(week_games["home_team"]) | set(week_games["away_team"]))
    print(f"\nPrecomputing elimination arrays for all {len(all_teams_playing)} teams playing Week {args.week}...")
    elimination_weeks = {team: team_elimination_week(sim, team) for team in all_teams_playing}

    recommendation = greedy_local_allocation(
        sim, all_teams_playing, n_entries=args.n_entries, elimination_weeks=elimination_weeks
    )
    print(f"\nRecommended allocation ({args.n_entries} entries, all {len(all_teams_playing)} teams considered):")
    print(f"  {recommendation.allocation}")
    print(f"  Expected payout: {recommendation.mean_payout:.2f} (SE {recommendation.standard_error:.2f})")

    top_candidates = sorted(current_week_survival, key=current_week_survival.get, reverse=True)[:TOP_N_FOR_SANITY_CHECK]
    top_candidates = [t for t in top_candidates if t in elimination_weeks]
    sanity_check = best_allocations(
        sim, top_candidates, n_entries=args.n_entries,
        elimination_weeks={t: elimination_weeks[t] for t in top_candidates},
    )[0]
    print(f"\nSanity check -- exhaustive search, top {len(top_candidates)} candidates by survival probability "
          f"({top_candidates}):")
    print(f"  {sanity_check.allocation}")
    print(f"  Expected payout: {sanity_check.mean_payout:.2f} (SE {sanity_check.standard_error:.2f})")

    greedy_payouts = score_allocation(sim, recommendation.allocation, elimination_weeks)
    exhaustive_payouts = score_allocation(sim, sanity_check.allocation, {t: elimination_weeks[t] for t in top_candidates})
    diff = greedy_payouts - exhaustive_payouts
    gap, gap_se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(args.n_paths))
    print(f"\nAll-candidate vs. top-{TOP_N_FOR_SANITY_CHECK} gap (paired): {gap:.2f}, SE {gap_se:.2f} "
          f"({'a real difference' if abs(gap) > 2 * gap_se else 'within noise at this path count -- consider them tied'})")

    out_dir = DEFAULT_STORE_ROOT / "pick_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    sheet = pd.DataFrame(
        [{"team": team, "entries": count} for team, count in recommendation.allocation.items()]
    )
    sheet.to_csv(out_dir / f"{args.year}_week{args.week}_{timestamp}.csv", index=False)
    sheet.to_csv(out_dir / "latest.csv", index=False)
    print(f"\nSaved pick sheet to {out_dir / f'{args.year}_week{args.week}_{timestamp}.csv'}")


if __name__ == "__main__":
    main()
