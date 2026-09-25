"""Pull and store the full Weeks 4-18 schedule from ESPN's public endpoint.

Free and keyless -- unrelated to the odds API's quota, safe to rerun.
Run directly: .venv/bin/python scripts/fetch_schedule.py
"""

from pathlib import Path

from survivor.data.schedule_client import compute_bye_teams, fetch_full_schedule
from survivor.data.storage import DEFAULT_STORE_ROOT

SEASON_YEAR = 2026
WEEKS = range(4, 19)


def main() -> None:
    schedule = fetch_full_schedule(WEEKS, SEASON_YEAR)

    out_dir = DEFAULT_STORE_ROOT / "schedule"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"schedule_weeks_{WEEKS.start}_{WEEKS.stop - 1}.csv"
    schedule.to_csv(out_path, index=False)

    print(f"Fetched {len(schedule)} games across weeks {WEEKS.start}-{WEEKS.stop - 1}.")
    print(f"Saved to {out_path}\n")

    print("Games and bye teams per week:")
    for week in WEEKS:
        week_games = schedule[schedule["week"] == week]
        byes = compute_bye_teams(week_games)
        bye_str = ", ".join(byes) if byes else "none"
        print(f"  Week {week:2d}: {len(week_games):2d} games, byes: {bye_str}")


if __name__ == "__main__":
    main()
