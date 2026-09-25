import pytest

from survivor.data.rival_tracker import pick_distribution, record_pick, used_teams


@pytest.fixture
def picks_path(tmp_path):
    return tmp_path / "picks.csv"


def test_record_and_query_used_teams(picks_path):
    record_pick("entry_1", 4, "BUF", "2026-10-04T13:00:00Z", path=picks_path)
    record_pick("entry_1", 5, "KC", "2026-10-11T13:00:00Z", path=picks_path)
    assert used_teams("entry_1", path=picks_path) == {"BUF", "KC"}


def test_used_teams_empty_for_unknown_entry(picks_path):
    assert used_teams("nobody", path=picks_path) == set()


def test_record_pick_normalizes_team_alias(picks_path):
    record_pick("entry_1", 4, "WSH", "2026-10-04T13:00:00Z", path=picks_path)
    assert used_teams("entry_1", path=picks_path) == {"WAS"}


def test_duplicate_pick_same_week_raises(picks_path):
    record_pick("entry_1", 4, "BUF", "2026-10-04T13:00:00Z", path=picks_path)
    with pytest.raises(ValueError):
        record_pick("entry_1", 4, "NE", "2026-10-04T13:00:00Z", path=picks_path)


def test_pick_distribution_counts_by_team(picks_path):
    record_pick("entry_1", 4, "BUF", "2026-10-04T13:00:00Z", path=picks_path)
    record_pick("entry_2", 4, "BUF", "2026-10-04T13:00:00Z", path=picks_path)
    record_pick("entry_3", 4, "KC", "2026-10-04T13:00:00Z", path=picks_path)

    dist = pick_distribution(4, path=picks_path)
    assert dist["BUF"] == 2
    assert dist["KC"] == 1


def test_different_entries_can_pick_same_team_same_week(picks_path):
    record_pick("entry_1", 4, "BUF", "2026-10-04T13:00:00Z", path=picks_path)
    record_pick("entry_2", 4, "BUF", "2026-10-04T13:00:00Z", path=picks_path)
    assert used_teams("entry_1", path=picks_path) == {"BUF"}
    assert used_teams("entry_2", path=picks_path) == {"BUF"}
