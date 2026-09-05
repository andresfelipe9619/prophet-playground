"""Pre-registration of predictions — lottery/analysis/registry.py.

Everything retrospective in this project can be adjusted after the fact: a
window moved, a model swapped, a run that "does not count". The registry is
the one surface that cannot, and its whole value is in what it *refuses*. So
that is what these tests are about — the guards, not the happy path.
"""

import pandas as pd
import pytest

from lottery.analysis.registry import (
    COLUMNS,
    RegistryError,
    load,
    pending,
    record,
    record_predictions,
    score_pending,
    status,
    summary,
)
from lottery.analysis.tickets import Ticket
from lottery.utils.sample_data import load_sample_and_preprocess

TICKET = Ticket(main=(3, 12, 19, 27, 41), super_ball=8)
OTHER = Ticket(main=(1, 5, 9, 14, 22), super_ball=16)
FUTURE = "2099-01-04"


@pytest.fixture
def registry_path(tmp_path):
    return str(tmp_path / "predictions.csv")


# ------------------------------------------------------------ the refusals

def test_a_draw_already_in_the_past_is_refused(registry_path):
    """A prediction recorded after its draw proves nothing."""
    with pytest.raises(RegistryError, match="not in the future"):
        record(TICKET, "2020-01-04", "late", path=registry_path)


def test_today_is_not_in_the_future_either(registry_path):
    """The draw happens on the day; same-day registration is unverifiable."""
    now = pd.Timestamp("2026-05-10T12:00:00+00:00").to_pydatetime()
    with pytest.raises(RegistryError, match="not in the future"):
        record(TICKET, "2026-05-10", "same day", path=registry_path, now=now)


def test_tomorrow_is_accepted(registry_path):
    now = pd.Timestamp("2026-05-10T12:00:00+00:00").to_pydatetime()
    row = record(TICKET, "2026-05-11", "tomorrow", path=registry_path, now=now)
    assert row["label"] == "tomorrow"


def test_a_second_prediction_for_the_same_draw_and_label_is_refused(registry_path):
    record(TICKET, FUTURE, "Prophet", path=registry_path)
    with pytest.raises(RegistryError, match="already on the record"):
        record(OTHER, FUTURE, "Prophet", path=registry_path)


def test_the_refusal_names_what_is_already_registered(registry_path):
    record(TICKET, FUTURE, "Prophet", path=registry_path)
    with pytest.raises(RegistryError, match=r"3-12-19-27-41"):
        record(OTHER, FUTURE, "Prophet", path=registry_path)


def test_a_revision_under_a_different_label_is_allowed(registry_path):
    """Append-only: both stay visible rather than one overwriting the other."""
    record(TICKET, FUTURE, "Prophet", path=registry_path)
    record(OTHER, FUTURE, "Prophet v2", path=registry_path)
    assert len(load(registry_path)) == 2


def test_the_same_label_may_predict_different_draws(registry_path):
    record(TICKET, FUTURE, "Prophet", path=registry_path)
    record(OTHER, "2099-01-06", "Prophet", path=registry_path)
    assert len(load(registry_path)) == 2


def test_only_a_ticket_may_be_registered(registry_path):
    with pytest.raises(TypeError, match="must be a Ticket"):
        record((3, 12, 19, 27, 41), FUTURE, "raw tuple", path=registry_path)


# ---------------------------------------------------------------- the file

def test_an_absent_registry_loads_as_an_empty_frame_with_the_right_columns(registry_path):
    registry = load(registry_path)
    assert registry.empty
    assert list(registry.columns) == COLUMNS


def test_a_recorded_row_is_pending_and_unscored(registry_path):
    record(TICKET, FUTURE, "Prophet", path=registry_path)
    registry = load(registry_path)
    assert len(registry) == 1
    assert pd.isna(registry.iloc[0]["scored_at"])
    assert pd.isna(registry.iloc[0]["main_matches"])
    assert len(pending(registry_path)) == 1


def test_main_numbers_are_stored_sorted(registry_path):
    record(Ticket(main=(41, 3, 27, 12, 19), super_ball=8), FUTURE, "shuffled", path=registry_path)
    assert load(registry_path).iloc[0]["main"] == "3-12-19-27-41"


def test_recording_survives_a_reload(registry_path):
    record(TICKET, FUTURE, "a", path=registry_path)
    record(OTHER, "2099-02-01", "b", path=registry_path)
    labels = set(load(registry_path)["label"])
    assert labels == {"a", "b"}


def test_record_predictions_accepts_a_models_raw_output(registry_path):
    """Collisions are filled at random, so the ticket may differ from the prediction."""
    import numpy as np
    row = record_predictions({0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 3}, FUTURE, "collided",
                             rng=np.random.default_rng(0), path=registry_path)
    assert len(set(row["main"].split("-"))) == 5
    assert row["super_ball"] == 3


# --------------------------------------------------------------- scoring

@pytest.fixture
def history():
    return load_sample_and_preprocess(n_draws=200)


def _register_against(df, balls, registry_path, n=3):
    """Register a prediction for each of the last n draws, dated before they happened."""
    for i in range(1, n + 1):
        draw_date = df["ds"].iloc[-i]
        before = (draw_date - pd.Timedelta(days=1)).to_pydatetime()
        record(TICKET, draw_date, f"model-{i}", path=registry_path, now=before)


def test_scoring_fills_in_every_draw_that_has_happened(history, registry_path):
    df, balls = history
    _register_against(df, balls, registry_path, n=3)

    scored = score_pending(df, balls, path=registry_path)
    assert scored["main_matches"].notna().all()
    assert scored["scored_at"].notna().all()
    assert len(pending(registry_path)) == 0


def test_scoring_is_all_or_nothing_never_a_subset(history, registry_path):
    """Picking which predictions to count is the failure this module prevents."""
    df, balls = history
    _register_against(df, balls, registry_path, n=3)
    record(TICKET, FUTURE, "not yet", path=registry_path)

    scored = score_pending(df, balls, path=registry_path)
    assert scored["main_matches"].notna().sum() == 3, "every eligible row"
    assert len(pending(registry_path)) == 1, "and only the future one left pending"


def test_a_future_draw_is_never_scored(history, registry_path):
    df, balls = history
    record(TICKET, FUTURE, "future", path=registry_path)
    scored = score_pending(df, balls, path=registry_path)
    assert scored["main_matches"].isna().all()


def test_rescoring_cannot_rewrite_history(history, registry_path):
    """Re-running is safe: already-scored rows are left untouched."""
    df, balls = history
    _register_against(df, balls, registry_path, n=2)

    first = score_pending(df, balls, path=registry_path)
    stamps = list(first["scored_at"])
    again = score_pending(df, balls, path=registry_path)
    assert list(again["scored_at"]) == stamps
    assert list(again["main_matches"]) == list(first["main_matches"])


def test_the_score_matches_what_was_actually_drawn(history, registry_path):
    df, balls = history
    draw_date = df["ds"].iloc[-1]
    record(TICKET, draw_date, "m", path=registry_path,
           now=(draw_date - pd.Timedelta(days=1)).to_pydatetime())

    row = score_pending(df, balls, path=registry_path).iloc[0]
    drawn = set(int(v) for v in balls.iloc[-1][:5])
    assert row["main_matches"] == len(set(TICKET.main) & drawn)
    assert row["actual_super"] == int(balls.iloc[-1][5])


def test_scoring_an_empty_registry_is_a_no_op(history, registry_path):
    df, balls = history
    assert score_pending(df, balls, path=registry_path).empty


# --------------------------------------------------------------- reporting

def test_summary_of_an_empty_registry_has_the_agreed_columns(registry_path):
    table = summary(path=registry_path)
    assert table.empty
    assert "min_detectable_effect" in table.columns


def test_summary_reports_the_resolution_beside_the_verdict(history, registry_path):
    """A young registry cannot say much and has to say so."""
    df, balls = history
    _register_against(df, balls, registry_path, n=3)
    score_pending(df, balls, path=registry_path)

    table = summary(path=registry_path)
    assert len(table) == 1
    assert table.iloc[0]["n_scored"] == 3
    assert {"effect", "ci_low", "ci_high", "min_detectable_effect",
            "p_value_better_than_chance"} <= set(table.columns)
    assert table.iloc[0]["min_detectable_effect"] > 1.0, "3 predictions can detect almost nothing"
    assert table.iloc[0]["ci_low"] < table.iloc[0]["ci_high"]


def test_summary_can_split_by_label(history, registry_path):
    df, balls = history
    _register_against(df, balls, registry_path, n=3)
    score_pending(df, balls, path=registry_path)
    assert len(summary(path=registry_path, by_label=True)) == 3


def test_status_counts_recorded_scored_and_pending(history, registry_path):
    df, balls = history
    _register_against(df, balls, registry_path, n=2)
    record(TICKET, FUTURE, "future", path=registry_path)
    score_pending(df, balls, path=registry_path)

    state = status(registry_path)
    assert state["n_recorded"] == 3
    assert state["n_scored"] == 2
    assert state["n_pending"] == 1
    assert state["next_draw"] == pd.Timestamp(FUTURE)


def test_status_of_an_empty_registry_reports_no_resolution(registry_path):
    import numpy as np
    state = status(registry_path)
    assert state["n_recorded"] == 0
    assert np.isnan(state["min_detectable_effect"])
