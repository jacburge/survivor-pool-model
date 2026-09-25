"""Manual connectivity check: one live pull against The Odds API.

Not part of the automated test suite (avoids burning API quota on every
test run). Run directly: .venv/bin/python scripts/test_odds_pull.py
"""

from survivor.data.odds_client import OddsAPIClient, parse_odds_events
from survivor.data.storage import save_raw_pull


def main() -> None:
    client = OddsAPIClient()
    events = client.get_odds()
    path = save_raw_pull("the_odds_api_nfl", events)

    print(f"Fetched {len(events)} games. Quota remaining: {client.last_quota_remaining}, used: {client.last_quota_used}")
    print(f"Raw pull saved to {path}")

    df = parse_odds_events(events)
    if df.empty:
        print("No games returned (may be off-season for lines, or no games currently listed).")
        return

    books_per_game = df.groupby("game_id")["bookmaker"].nunique()
    print(f"Games with odds: {df['game_id'].nunique()}")
    print(f"Books per game -- min: {books_per_game.min()}, median: {books_per_game.median()}")
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
