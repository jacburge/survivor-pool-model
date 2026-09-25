"""One-time historical pull for fitting the popularity model (Phase 4).

Free, keyless source (SurvivorGrid) -- no relation to the odds API quota.
Paced at 1 request/second out of courtesy to an unofficial, undocumented
site with no published rate limit.

Run: .venv/bin/python scripts/collect_historical_pick_data.py
"""

from survivor.data.storage import DEFAULT_STORE_ROOT
from survivor.data.survivorgrid_client import fetch_historical_pick_grids

YEARS = range(2022, 2026)  # 2022-2025
WEEKS = range(1, 19)


def main() -> None:
    print(f"Pulling SurvivorGrid pick grids for {list(YEARS)}, weeks {WEEKS.start}-{WEEKS.stop - 1}...")
    data = fetch_historical_pick_grids(YEARS, WEEKS, delay_seconds=1.0)

    out_dir = DEFAULT_STORE_ROOT / "survivorgrid_historical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pick_grids_{YEARS.start}_{YEARS.stop - 1}.csv"
    data.to_csv(out_path, index=False)

    print(f"Fetched {len(data)} team-week rows across {data['year'].nunique()} seasons.")
    print(f"Saved to {out_path}")
    print(data.groupby("year")["week"].nunique())


if __name__ == "__main__":
    main()
