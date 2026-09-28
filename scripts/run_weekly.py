"""Weekly production run (Phase 7): refresh real data, recommend an allocation
of your entries across this week's candidate teams, and archive the result.

Pulls a fresh schedule (this week through Week 18, free/ESPN, always --
there's no quota reason to ever skip it) and live current-week odds (The
Odds API, uses quota). The odds pull is ON by default here, unlike
refresh_all.py's dev-focused default -- this script's entire purpose is a
real weekly decision, exactly the case the odds-API conservation guidance
carves out as worth spending quota on. Pass --skip-odds-refresh to reuse
data_store/odds/latest.csv instead (e.g. if you already refreshed today,
or --week is a past week the live API no longer covers -- see below).

Backtesting a past week: the Odds API only returns current/upcoming games,
so --week for an already-played week needs --skip-odds-refresh plus
data_store/odds/latest.csv actually containing that week's *pre-game*
odds (i.e. pulled before it was played -- check the file's own pull
timestamp against the week's kickoff before trusting this, or you'd be
silently feeding the model a stale snapshot from some other week instead).

Fits current-week ratings from real spreads (Phase 3), gets real
current-week survival probabilities via Shin devig (Phase 2), runs the
field simulator through Week 18 (Phase 5), then recommends a per-entry
allocation via greedy_local_entry_allocation over every team playing this
week (Phase 6). Also reports the same search restricted to the top 5
candidates by survival probability as a sanity cross-check, using the same
simulated paths so the comparison is paired, not independent -- the
all-candidate search meaningfully beat that restricted one on real Week 3
and Week 4, 2026 data (plan.md's Phase 6 section has the numbers).

Entry tracking (survivor.data.my_entries): entries are named entry_1..
entry_N (N = --n-entries) and their history is read from
data_store/my_entries/. Nothing recorded yet (true before Week 4) means
every entry is alive with an empty used-teams set, which the allocation
search handles like any other state, not a special case -- verified this
reduces to exactly the same result greedy_local_allocation would give,
with no runtime penalty, since team_elimination_week's cache collapses
identical (empty) histories to one computation. Pass --record to commit
this run's recommendation into the tracker as each entry's pick for
--week; without it, this is a look, not a commitment. Marking who actually
won or lost each week (record_result) is a separate, manual step for now.

Runtime scales with --n-paths: precomputing elimination arrays for every
team playing (needed for the all-candidate search) is the added cost
beyond Phase 5's own validated field-simulation runtime, since it's one
Hungarian-assignment solve per (used-teams history, team) pair -- same as
per-team once entries share a history, as they do for a fresh --week 4.
Rough budget at 500 rivals, ~32 teams playing, fresh entries: 20,000 paths
~9 minutes, 80,000 paths ~35-40 minutes (over Phase 5's 30-minute bar) --
use a lower --n-paths for a quick look and raise it for the final
pre-lock run. Diverged histories cost more (up to one solve per distinct
history per team, not one total).

Rating fit uses this week's real spreads plus every earlier week's real
closing spreads this season (survivor.data.survivorgrid_client.
fetch_season_to_date_games, free/keyless -- one extra request per prior
week, paced at 1/sec) rather than this week's spreads alone: fitting on
just one week is severely underdetermined (16 games informing 32 teams'
ratings), and that noise otherwise carries straight into the future-week
projections the rollout depends on. Still an open item even with this fix
-- see DEFAULT_SEASON_TO_DATE_RIDGE's comment in ratings.py and
plan.md's Phase 3 section for the real numbers and what's still unresolved.

Run: .venv/bin/python scripts/run_weekly.py --week 4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from survivor.data import my_entries, schedule_client
from survivor.data.odds_client import OddsAPIClient, parse_odds_events
from survivor.data.storage import DEFAULT_STORE_ROOT, save_raw_pull
from survivor.data.survivorgrid_client import fetch_season_to_date_games
from survivor.decision.portfolio import greedy_local_entry_allocation, score_entries
from survivor.probability.current_week import compute_current_week_probabilities, compute_current_week_spreads
from survivor.probability.ratings import DEFAULT_SEASON_TO_DATE_RIDGE, fit_team_ratings
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


def load_cached_odds() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_STORE_ROOT / "odds" / "latest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, required=True, help="the current week to decide for")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--n-entries", type=int, default=10)
    parser.add_argument("--n-rivals", type=int, default=500, help="assumed field size -- unknown until lock day")
    parser.add_argument("--n-paths", type=int, default=20000, help="see docstring for the runtime-vs-precision tradeoff")
    parser.add_argument("--pot", type=float, default=9000.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-odds-refresh", action="store_true",
                         help="reuse data_store/odds/latest.csv instead of a live API call "
                              "(required for a past --week; see docstring)")
    parser.add_argument("--record", action="store_true",
                         help="commit this run's recommendation into data_store/my_entries/ "
                              "as each entry's pick for --week (default: just look, don't commit)")
    args = parser.parse_args()

    print(f"Refreshing schedule (weeks {args.week}-{FINAL_WEEK}, {args.year})...")
    schedule = refresh_schedule(args.year, args.week)

    if args.skip_odds_refresh:
        print("Skipping odds refresh, using cached data_store/odds/latest.csv as-is.")
        odds = load_cached_odds()
    else:
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

    if args.week > 1:
        print(f"\nPulling Weeks 1-{args.week - 1} real closing spreads from SurvivorGrid for the rating fit "
              f"({args.week - 1} requests, free/keyless, paced 1/sec)...")
        season_to_date = fetch_season_to_date_games(args.year, through_week=args.week - 1)
    else:
        season_to_date = pd.DataFrame(columns=["week", "home_team", "away_team", "home_spread"])

    fit_games = pd.concat(
        [season_to_date[["home_team", "away_team", "home_spread"]], week_spreads[["home_team", "away_team", "home_spread"]]],
        ignore_index=True,
    )
    fit = fit_team_ratings(fit_games, ridge=DEFAULT_SEASON_TO_DATE_RIDGE)
    print(f"\nFitted ratings from {len(fit_games)} games (Weeks 1-{args.week}, real spreads) "
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

    entry_ids = [f"entry_{i + 1}" for i in range(args.n_entries)]
    alive_ids = my_entries.alive_entries(entry_ids)
    if not alive_ids:
        raise SystemExit(f"No alive entries among {entry_ids} -- nothing to recommend.")
    if len(alive_ids) < len(entry_ids):
        print(f"\n{len(entry_ids) - len(alive_ids)} of {len(entry_ids)} entries already eliminated: "
              f"{sorted(set(entry_ids) - set(alive_ids))}")
    used_teams_by_entry = my_entries.used_teams_by_entry(alive_ids)

    all_teams_playing = sorted(set(week_games["home_team"]) | set(week_games["away_team"]))
    print(f"\nRecommending picks for {len(alive_ids)} alive entries across all "
          f"{len(all_teams_playing)} teams playing Week {args.week}...")
    recommendation = greedy_local_entry_allocation(sim, used_teams_by_entry, all_teams_playing)

    team_counts: dict[str, int] = {}
    for team in recommendation.values():
        team_counts[team] = team_counts.get(team, 0) + 1
    recommendation_arrays = {
        entry_id: team_elimination_week(sim, team, used_teams_by_entry[entry_id])
        for entry_id, team in recommendation.items()
    }
    recommendation_payout = score_entries(sim, recommendation_arrays)
    print(f"\nRecommended allocation ({team_counts}):")
    for entry_id, team in sorted(recommendation.items()):
        history = used_teams_by_entry[entry_id]
        print(f"  {entry_id} -> {team}" + (f"  (already used: {sorted(history)})" if history else ""))
    print(f"  Expected total payout: {recommendation_payout.mean():.2f} "
          f"(SE {recommendation_payout.std(ddof=1) / np.sqrt(args.n_paths):.2f})")

    top_candidates = sorted(current_week_survival, key=current_week_survival.get, reverse=True)[:TOP_N_FOR_SANITY_CHECK]
    top_candidates = [t for t in top_candidates if t in all_teams_playing]
    sanity_check = greedy_local_entry_allocation(sim, used_teams_by_entry, top_candidates)
    sanity_arrays = {
        entry_id: team_elimination_week(sim, team, used_teams_by_entry[entry_id])
        for entry_id, team in sanity_check.items()
    }
    sanity_payout = score_entries(sim, sanity_arrays)
    print(f"\nSanity check -- same search, restricted to the top {len(top_candidates)} candidates by survival "
          f"probability ({top_candidates}):")
    print(f"  Expected total payout: {sanity_payout.mean():.2f} "
          f"(SE {sanity_payout.std(ddof=1) / np.sqrt(args.n_paths):.2f})")

    diff = recommendation_payout - sanity_payout
    gap, gap_se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(args.n_paths))
    print(f"\nAll-candidate vs. top-{TOP_N_FOR_SANITY_CHECK} gap (paired): {gap:.2f}, SE {gap_se:.2f} "
          f"({'a real difference' if abs(gap) > 2 * gap_se else 'within noise at this path count -- consider them tied'})")

    out_dir = DEFAULT_STORE_ROOT / "pick_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    sheet = pd.DataFrame(
        [
            {"entry_id": entry_id, "team": team, "used_teams_before": ",".join(sorted(used_teams_by_entry[entry_id]))}
            for entry_id, team in sorted(recommendation.items())
        ]
    )
    sheet.to_csv(out_dir / f"{args.year}_week{args.week}_{timestamp}.csv", index=False)
    sheet.to_csv(out_dir / "latest.csv", index=False)
    print(f"\nSaved pick sheet to {out_dir / f'{args.year}_week{args.week}_{timestamp}.csv'}")

    if args.record:
        for entry_id, team in recommendation.items():
            my_entries.record_pick(entry_id, args.week, team)
        print(f"Recorded {len(recommendation)} picks into data_store/my_entries/ for Week {args.week}.")
    else:
        print("Not recorded (pass --record to commit this as each entry's pick for the week).")


if __name__ == "__main__":
    main()
