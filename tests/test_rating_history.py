import pytest

from survivor.data.rating_history import load_rating_history, record_weekly_ratings, to_wide_by_season


@pytest.fixture
def history_path(tmp_path):
    return tmp_path / "history.csv"


def test_record_and_load_round_trips(history_path):
    record_weekly_ratings("survivorgrid_2024", 2024, 1, {"BUF": 3.0, "NE": -2.0}, 2.5, path=history_path)
    history = load_rating_history(history_path)
    assert len(history) == 2
    assert set(history["team"]) == {"BUF", "NE"}
    assert history[history["team"] == "BUF"]["rating"].iloc[0] == pytest.approx(3.0)


def test_load_missing_file_returns_empty(history_path):
    history = load_rating_history(history_path)
    assert len(history) == 0


def test_rerecording_same_week_replaces_not_duplicates(history_path):
    record_weekly_ratings("survivorgrid_2024", 2024, 1, {"BUF": 3.0}, 2.5, path=history_path)
    record_weekly_ratings("survivorgrid_2024", 2024, 1, {"BUF": 3.5}, 2.6, path=history_path)  # refit, revised
    history = load_rating_history(history_path)
    assert len(history) == 1
    assert history.iloc[0]["rating"] == pytest.approx(3.5)


def test_different_weeks_accumulate(history_path):
    record_weekly_ratings("survivorgrid_2024", 2024, 1, {"BUF": 3.0}, 2.5, path=history_path)
    record_weekly_ratings("survivorgrid_2024", 2024, 2, {"BUF": 3.2}, 2.4, path=history_path)
    history = load_rating_history(history_path)
    assert len(history) == 2
    assert set(history["week"]) == {1, 2}


def test_different_sources_and_seasons_kept_separate(history_path):
    record_weekly_ratings("survivorgrid_2023", 2023, 1, {"BUF": 1.0}, 2.0, path=history_path)
    record_weekly_ratings("survivorgrid_2024", 2024, 1, {"BUF": 5.0}, 2.0, path=history_path)
    history = load_rating_history(history_path)
    assert len(history) == 2


def test_to_wide_by_season_shapes_for_fit_weekly_rating_std(history_path):
    record_weekly_ratings("survivorgrid_2023", 2023, 1, {"BUF": 1.0, "NE": -1.0}, 2.0, path=history_path)
    record_weekly_ratings("survivorgrid_2023", 2023, 2, {"BUF": 1.5, "NE": -1.5}, 2.1, path=history_path)
    record_weekly_ratings("survivorgrid_2024", 2024, 1, {"BUF": 3.0, "NE": -3.0}, 2.0, path=history_path)

    history = load_rating_history(history_path)
    wide_frames = to_wide_by_season(history)

    assert len(wide_frames) == 2  # one per (source, year)
    sizes = sorted(len(w) for w in wide_frames)
    assert sizes == [1, 2]  # 2024 has 1 week recorded, 2023 has 2
    for wide in wide_frames:
        assert set(wide.columns) == {"BUF", "NE"}
        assert list(wide.index) == sorted(wide.index)  # sorted by week
