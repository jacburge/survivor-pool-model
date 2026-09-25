"""One-command refresh of every Phase 1 data source.

Schedule (ESPN) and pick percentages (SurvivorGrid) are free and keyless,
so they refresh by default. Odds (The Odds API) is quota-limited, so it
only refreshes when --odds is passed -- don't burn the free tier's 500
requests/month on routine refreshes.

Every source's raw payload for this run is archived via
survivor.data.storage.save_raw_pull (timestamped, under data_store/, never
committed). Each source also gets a data_store/<source>/latest.csv
convenience snapshot for downstream phases to load without hunting through
timestamps.

Usage:
  .venv/bin/python scripts/refresh_all.py                  # schedule + pick percentages
  .venv/bin/python scripts/refresh_all.py --odds            # + a live odds pull
  .venv/bin/python scripts/refresh_all.py --pick-week 5
"""

from __future__ import annotations

import argparse

import pandas as pd
import requests

from survivor.data import schedule_client, survivorgrid_client
from survivor.data.odds_client import OddsAPIClient, parse_odds_events
from survivor.data.storage import DEFAULT_STORE_ROOT, save_raw_pull

MIN_BOOKS_PER_GAME = 3


def refresh_schedule(year: int, weeks: range) -> pd.DataFrame:
    raw_by_week = {week: schedule_client.fetch_week_schedule(week, year) for week in weeks}
    save_raw_pull("schedule", raw_by_week)

    parsed = pd.concat(
        [schedule_client.parse_week_schedule(raw, week) for week, raw in raw_by_week.items()],
        ignore_index=True,
    )
    return schedule_client.add_effective_lock_times(parsed)


def refresh_pick_percentages(year: int, week: int) -> pd.DataFrame:
    df = survivorgrid_client.fetch_pick_grid(year, week)
    save_raw_pull("survivorgrid", df.to_dict(orient="records"))
    return df


def refresh_odds(markets: str) -> tuple[pd.DataFrame, int | None, int | None]:
    client = OddsAPIClient()
    events = client.get_odds(markets=markets)
    save_raw_pull("the_odds_api_nfl", events)
    return parse_odds_events(events), client.last_quota_remaining, client.last_quota_used


def _write_latest(df: pd.DataFrame, source: str) -> None:
    out_dir = DEFAULT_STORE_ROOT / source
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "latest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--week-start", type=int, default=4)
    parser.add_argument("--week-end", type=int, default=18)
    parser.add_argument("--pick-week", type=int, default=4, help="week to pull SurvivorGrid's pick distribution for")
    parser.add_argument("--odds", action="store_true", help="also refresh live odds (uses API quota)")
    parser.add_argument("--odds-markets", default="h2h,spreads")
    args = parser.parse_args()

    weeks = range(args.week_start, args.week_end + 1)

    print(f"Refreshing schedule (weeks {weeks.start}-{weeks.stop - 1}, {args.year})...")
    schedule = refresh_schedule(args.year, weeks)
    _write_latest(schedule, "schedule")
    print(f"  {len(schedule)} games. Saved to data_store/schedule/latest.csv")

    print(f"\nRefreshing pick percentages (week {args.pick_week}, {args.year})...")
    try:
        picks = refresh_pick_percentages(args.year, args.pick_week)
    except requests.HTTPError as exc:
        print(f"  SurvivorGrid hasn't published week {args.pick_week} yet ({exc.response.status_code}): "
              "expected before that week is current -- try again closer to it.")
    else:
        _write_latest(picks, "survivorgrid")
        print(f"  {len(picks)} teams. Saved to data_store/survivorgrid/latest.csv")

    if args.odds:
        print("\nRefreshing odds (live API call)...")
        odds, quota_remaining, quota_used = refresh_odds(args.odds_markets)
        _write_latest(odds, "odds")
        print(f"  {odds['game_id'].nunique()} games. Saved to data_store/odds/latest.csv")
        print(f"  API quota -- used: {quota_used}, remaining: {quota_remaining}")

        books_per_game = odds.groupby("game_id")["bookmaker"].nunique()
        under_minimum = books_per_game[books_per_game < MIN_BOOKS_PER_GAME]
        if under_minimum.empty:
            print(f"  All games have at least {MIN_BOOKS_PER_GAME} books.")
        else:
            print(f"  WARNING: {len(under_minimum)} game(s) have fewer than {MIN_BOOKS_PER_GAME} books.")
    else:
        print("\nSkipping odds refresh (pass --odds to include a live pull).")

    print("\nDone. All team names normalized through survivor.data.team_keys -- "
          "any unrecognized team would have raised a KeyError above rather than silently mismatching.")


if __name__ == "__main__":
    main()
