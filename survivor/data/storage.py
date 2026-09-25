"""Timestamped, append-only storage for raw pulls.

Every ingestion run writes its raw response alongside a UTC timestamp so
later runs are reproducible and comparable (e.g. "how did lines move
between Thursday kickoff and the Sunday lock"). Nothing here is committed
to git -- data_store/ is gitignored.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STORE_ROOT = Path(__file__).resolve().parents[2] / "data_store"


def save_raw_pull(source: str, payload: Any, store_root: Path = DEFAULT_STORE_ROOT) -> Path:
    """Write a timestamped raw JSON payload under data_store/<source>/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    directory = store_root / source
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_latest_pull(source: str, store_root: Path = DEFAULT_STORE_ROOT) -> tuple[Path, Any]:
    """Load the most recent stored pull for a source, by filename (timestamps sort lexically)."""
    directory = store_root / source
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no stored pulls found for source {source!r} in {directory}")
    latest = files[-1]
    return latest, json.loads(latest.read_text())
