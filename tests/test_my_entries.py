import pytest

from survivor.data.my_entries import (
    alive_entries,
    is_alive,
    record_allocation,
    record_pick,
    record_result,
    used_teams,
    used_teams_by_entry,
)


@pytest.fixture
def picks_path(tmp_path):
    return tmp_path / "picks.csv"


def test_fresh_entry_with_no_history_is_alive(picks_path):
    assert is_alive("entry_1", path=picks_path)


def test_fresh_entry_with_no_history_has_no_used_teams(picks_path):
    assert used_teams("entry_1", path=picks_path) == set()


def test_record_pick_and_query_used_teams(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_pick("entry_1", 5, "KC", path=picks_path)
    assert used_teams("entry_1", path=picks_path) == {"BUF", "KC"}


def test_record_pick_normalizes_team_alias(picks_path):
    record_pick("entry_1", 4, "WSH", path=picks_path)
    assert used_teams("entry_1", path=picks_path) == {"WAS"}


def test_rerecording_same_week_replaces_not_duplicates(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_pick("entry_1", 4, "KC", path=picks_path)  # changed my mind before lock
    assert used_teams("entry_1", path=picks_path) == {"KC"}


def test_pending_pick_is_still_alive(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    assert is_alive("entry_1", path=picks_path)


def test_winning_pick_stays_alive(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_result("entry_1", 4, survived=True, path=picks_path)
    assert is_alive("entry_1", path=picks_path)


def test_losing_pick_is_eliminated(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_result("entry_1", 4, survived=False, path=picks_path)
    assert not is_alive("entry_1", path=picks_path)


def test_eliminated_entry_stays_eliminated_even_with_a_later_pending_pick(picks_path):
    # shouldn't happen in practice (a dead entry can't keep picking), but
    # is_alive must not be fooled by a later unresolved row
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_result("entry_1", 4, survived=False, path=picks_path)
    record_pick("entry_1", 5, "KC", path=picks_path)
    assert not is_alive("entry_1", path=picks_path)


def test_record_result_without_a_prior_pick_raises(picks_path):
    with pytest.raises(ValueError):
        record_result("entry_1", 4, survived=True, path=picks_path)


def test_alive_entries_filters_out_eliminated_ones(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_result("entry_1", 4, survived=False, path=picks_path)
    record_pick("entry_2", 4, "KC", path=picks_path)
    record_result("entry_2", 4, survived=True, path=picks_path)

    assert alive_entries(["entry_1", "entry_2", "entry_3"], path=picks_path) == ["entry_2", "entry_3"]


def test_used_teams_by_entry_covers_each_requested_entry(picks_path):
    record_pick("entry_1", 4, "BUF", path=picks_path)
    record_pick("entry_2", 4, "KC", path=picks_path)
    result = used_teams_by_entry(["entry_1", "entry_2", "entry_3"], path=picks_path)
    assert result == {"entry_1": {"BUF"}, "entry_2": {"KC"}, "entry_3": set()}


def test_record_allocation_assigns_and_records_every_entry(picks_path):
    allocation = {"BUF": 2, "KC": 1}
    assignment = record_allocation(allocation, week=4, entry_ids=["e1", "e2", "e3"], path=picks_path)

    assert set(assignment.values()) == {"BUF", "KC"}
    assert list(assignment) == ["e1", "e2", "e3"]
    for entry_id, team in assignment.items():
        assert used_teams(entry_id, path=picks_path) == {team}


def test_record_allocation_mismatched_entry_count_raises(picks_path):
    with pytest.raises(ValueError):
        record_allocation({"BUF": 2, "KC": 1}, week=4, entry_ids=["e1", "e2"], path=picks_path)
