"""Load Megan's weekly-job output into the database.

run_weekly.py already writes two files per run, unchanged by anything
here:
  data_store/pick_sheets/latest.csv   -- entry_id, team, used_teams_before
  data_store/my_entries/picks.csv     -- entry_id, week, team, survived

This script is the bridge: read those, upsert into `recommendations` and
`picks`. It's what a cron job calls right after run_weekly.py finishes --
the weekly job's own code never needs to know the database exists.

Also read: data_store/pick_sheets/latest_summary.json -- expected payout, SE,
objective and win probability, written by run_weekly.py.

Run: python ingest.py --league-id 1 --week 4 --store-root /path/to/data_store
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from models import Entry, Pick, Recommendation, SessionLocal, init_db


def ingest_recommendation(db, league_id: int, week: int, pick_sheet_path: Path,
                           expected_payout: float | None, se: float | None,
                           objective: str | None = None, win_probability: float | None = None) -> int:
    if not pick_sheet_path.exists():
        print(f"  no pick sheet at {pick_sheet_path}, skipping recommendation ingest")
        return 0
    n = 0
    with open(pick_sheet_path) as f:
        for row in csv.DictReader(f):
            existing = (
                db.query(Recommendation)
                .filter_by(league_id=league_id, week=week, entry_id=row["entry_id"])
                .one_or_none()
            )
            if existing is None:
                existing = Recommendation(league_id=league_id, week=week, entry_id=row["entry_id"])
                db.add(existing)
            existing.team = row["team"]
            existing.used_teams_before = row.get("used_teams_before", "")
            existing.expected_total_payout = expected_payout
            existing.payout_se = se
            existing.objective = objective
            existing.win_probability = win_probability
            n += 1
    return n


def ingest_picks(db, league_id: int, picks_csv_path: Path) -> int:
    if not picks_csv_path.exists():
        print(f"  no picks file at {picks_csv_path}, skipping picks ingest")
        return 0
    n = 0
    with open(picks_csv_path) as f:
        for row in csv.DictReader(f):
            entry = db.query(Entry).filter_by(league_id=league_id, entry_id=row["entry_id"]).one_or_none()
            if entry is None:
                print(f"  entry {row['entry_id']!r} not registered in league {league_id}, skipping")
                continue
            week = int(row["week"])
            existing = db.query(Pick).filter_by(entry_id_fk=entry.id, week=week).one_or_none()
            survived = {"True": True, "False": False}.get(row.get("survived", ""), None)
            if existing is None:
                existing = Pick(entry_id_fk=entry.id, week=week, team=row["team"], survived=survived)
                db.add(existing)
            # An existing pick is what was actually chosen (or entered on the picks page),
            # so the weekly job's recommendation never overwrites it.
            n += 1
    return n


def export_picks(db, league_id: int, picks_csv_path: Path, before_week: int | None = None) -> int:
    """Write the database's picks out in my_entries' CSV shape, so run_weekly.py
    sees each entry's real history. The job's disk starts empty every run, so
    without this it would treat every entry as fresh each week."""
    query = (
        db.query(Entry.entry_id, Pick.week, Pick.team, Pick.survived)
        .join(Pick, Pick.entry_id_fk == Entry.id)
        .filter(Entry.league_id == league_id)
    )
    if before_week is not None:
        # this week's own pick must not count as "already used" when planning this week
        query = query.filter(Pick.week < before_week)
    rows = query.order_by(Entry.entry_id, Pick.week).all()
    picks_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(picks_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entry_id", "week", "team", "survived"])
        for entry_id, week, team, survived in rows:
            writer.writerow([entry_id, week, team, "" if survived is None else survived])
    return len(rows)


def load_summary(pick_sheet_dir: Path, week: int) -> dict:
    """Read run_weekly.py's latest_summary.json. Refuses a summary for a different week, so a
    stale file from an earlier run can never be ingested as this week's numbers."""
    path = pick_sheet_dir / "latest_summary.json"
    if not path.exists():
        raise SystemExit(f"no run summary at {path} -- did run_weekly.py finish?")
    summary = json.loads(path.read_text())
    if summary.get("week") != week:
        raise SystemExit(f"{path} is for week {summary.get('week')}, expected week {week} -- stale file?")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--store-root", type=Path, required=True, help="path to survivor-pool-model/data_store")
    parser.add_argument("--payout", type=float, default=None, help="override the summary's expected payout")
    parser.add_argument("--se", type=float, default=None, help="override the summary's standard error")
    args = parser.parse_args()

    summary = load_summary(args.store_root / "pick_sheets", args.week)
    payout = args.payout if args.payout is not None else summary["expected_total_payout"]
    se = args.se if args.se is not None else summary["payout_se"]

    init_db()
    db = SessionLocal()
    try:
        n_rec = ingest_recommendation(
            db, args.league_id, args.week,
            args.store_root / "pick_sheets" / "latest.csv", payout, se,
            objective=summary.get("objective"), win_probability=summary.get("win_probability"),
        )
        n_pick = ingest_picks(db, args.league_id, args.store_root / "my_entries" / "picks.csv")
        db.commit()
        print(f"Ingested {n_rec} recommendation rows, {n_pick} pick rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
