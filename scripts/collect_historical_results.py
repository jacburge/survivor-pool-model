"""Refetch historical SurvivorGrid data including win/loss results, for validating
the field simulator's elimination curve against real history (Phase 5).

The first historical pull (collect_historical_pick_data.py) predates the
parser extracting the result field, so this refetches with it included.
Free, keyless source, paced at 1 request/second out of courtesy.

Run: .venv/bin/python scripts/collect_historical_results.py
"""

from survivor.data.storage import DEFAULT_STORE_ROOT
from survivor.data.survivorgrid_client import fetch_historical_pick_grids

YEARS = range(2023, 2026)  # 2023-2025
WEEKS = range(1, 19)


def main() -> None:
    print(f"Pulling SurvivorGrid pick grids (with results) for {list(YEARS)}, weeks {WEEKS.start}-{WEEKS.stop - 1}...")
    data = fetch_historical_pick_grids(YEARS, WEEKS, delay_seconds=1.0)

    out_dir = DEFAULT_STORE_ROOT / "survivorgrid_historical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pick_grids_with_results_{YEARS.start}_{YEARS.stop - 1}.csv"
    data.to_csv(out_path, index=False)

    print(f"Fetched {len(data)} team-week rows across {data['year'].nunique()} seasons.")
    print(f"Saved to {out_path}")
    print(data["result"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
