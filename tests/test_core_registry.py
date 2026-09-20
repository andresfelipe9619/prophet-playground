"""The three refusals, on a registry that knows about no domain at all.

`tests/test_registry.py` pins the same rules through the lottery's adapter and
passed unchanged through the lift, which is that refactor's proof. These pin
them on `core/` directly, so a future domain that gets them wrong fails here
rather than in whichever adapter happened to be written first.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from core.registry import (
    RegistryError,
    RegistrySchema,
    load,
    pending,
    record,
    score_pending,
    status,
)

SCHEMA = RegistrySchema(
    event_column="event_on",
    prediction_columns=("guess",),
    result_columns=("actual", "hit"),
)

TOMORROW = (datetime.now(UTC) + timedelta(days=2)).date().isoformat()
YESTERDAY = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "predictions.csv")


def _record(path, guess="A", event=TOMORROW, label="model", **kwargs):
    return record(SCHEMA, {"guess": guess}, event, label, path=path, **kwargs)


# ------------------------------------------------- refusal 1 · only the future


def test_an_event_that_already_happened_is_refused(path):
    with pytest.raises(RegistryError, match="not in the future"):
        _record(path, event=YESTERDAY)


def test_today_is_not_in_the_future_either(path):
    """The boundary. A prediction for a draw happening tonight is not evidence."""
    today = datetime.now(UTC).date().isoformat()
    with pytest.raises(RegistryError, match="not in the future"):
        _record(path, event=today)


def test_the_refusal_names_the_domains_own_column(path):
    """A reader of the error should not have to translate a generic name."""
    with pytest.raises(RegistryError, match="event_on"):
        _record(path, event=YESTERDAY)


def test_the_boundary_is_measured_against_the_injected_now(path):
    """`now` is injectable so the guard itself is testable, and for nothing else."""
    frozen = datetime(2030, 1, 10, tzinfo=UTC)
    with pytest.raises(RegistryError, match="not in the future"):
        _record(path, event="2030-01-10", now=frozen)
    row = _record(path, event="2030-01-11", now=frozen)
    assert row["event_on"] == pd.Timestamp("2030-01-11")


# ---------------------------------------------------- refusal 2 · append-only


def test_a_second_prediction_for_the_same_event_and_label_is_refused(path):
    _record(path)
    with pytest.raises(RegistryError, match="already on the record"):
        _record(path)


def test_the_refusal_shows_what_is_already_registered(path):
    """So the reader can see what they would have overwritten."""
    _record(path, guess="A")
    with pytest.raises(RegistryError, match="guess=A"):
        _record(path, guess="B")


def test_a_revision_under_a_different_label_is_allowed(path):
    """The sanctioned way to change your mind: both stay visible."""
    _record(path, guess="A", label="v1")
    _record(path, guess="B", label="v2")
    assert len(load(SCHEMA, path)) == 2


def test_the_same_label_may_predict_different_events(path):
    _record(path, event=TOMORROW, label="model")
    _record(path, event="2031-05-05", label="model")
    assert len(load(SCHEMA, path)) == 2


# ------------------------------------------------ refusal 3 · all, or none


def test_scoring_fills_in_every_resolvable_row(path):
    _record(path, guess="A", label="one")
    _record(path, guess="B", label="two")

    registry = score_pending(SCHEMA, path, lambda row: {"actual": "A", "hit": row["guess"] == "A"})
    assert registry["scored_at"].notna().all()
    assert list(registry["hit"]) == [True, False]


def test_a_row_the_domain_cannot_resolve_stays_pending(path):
    """None means "not yet", and is the only reason a row may be skipped."""
    _record(path, label="known")
    _record(path, label="unknown")

    registry = score_pending(SCHEMA, path,
                             lambda row: None if row["label"] == "unknown"
                             else {"actual": "A", "hit": True})
    assert int(registry["scored_at"].notna().sum()) == 1
    assert len(pending(SCHEMA, registry=registry)) == 1


def test_a_half_filled_result_is_refused(path):
    """A half-scored row reads as a scored one and would be counted as evidence."""
    _record(path)
    with pytest.raises(RegistryError, match="no value for"):
        score_pending(SCHEMA, path, lambda row: {"actual": "A"})


def test_rescoring_cannot_rewrite_history(path):
    _record(path)
    first = score_pending(SCHEMA, path, lambda row: {"actual": "A", "hit": True})
    again = score_pending(SCHEMA, path, lambda row: {"actual": "Z", "hit": False})

    assert list(again["actual"]) == list(first["actual"]) == ["A"]
    assert list(again["scored_at"]) == list(first["scored_at"])


def test_scoring_an_empty_registry_is_a_no_op(path):
    assert score_pending(SCHEMA, path, lambda row: {"actual": "A", "hit": True}).empty


# ------------------------------------------------------------ the shape


def test_an_absent_registry_loads_as_an_empty_frame_with_the_right_columns(path):
    registry = load(SCHEMA, path)
    assert registry.empty
    assert list(registry.columns) == list(SCHEMA.columns)


def test_the_column_order_puts_the_prediction_before_the_result(path):
    """So a committed file reads left to right as the story happened."""
    assert list(SCHEMA.columns) == [
        "recorded_at", "event_on", "label", "guess", "note", "scored_at", "actual", "hit"]


def test_a_prediction_missing_a_column_is_refused(path):
    """It would write a row that can never be scored."""
    with pytest.raises(RegistryError, match="missing"):
        record(SCHEMA, {}, TOMORROW, "model", path=path)


def test_a_prediction_with_an_extra_column_is_refused(path):
    """The caller thinks the schema says something it does not."""
    with pytest.raises(RegistryError, match="unexpected"):
        record(SCHEMA, {"guess": "A", "confidence": 0.9}, TOMORROW, "model", path=path)


def test_the_late_columns_survive_a_reload_as_objects(path):
    """An all-NA column pandas types as float64 makes the first real score an
    incompatible-dtype assignment: a FutureWarning today, an error later."""
    _record(path)
    registry = load(SCHEMA, path)
    for column in SCHEMA.late_columns:
        assert registry[column].dtype == object


def test_recording_survives_a_reload(path):
    _record(path, guess="A")
    assert load(SCHEMA, path).iloc[0]["guess"] == "A"


def test_status_counts_recorded_scored_and_pending(path):
    _record(path, label="one")
    _record(path, label="two")
    score_pending(SCHEMA, path,
                  lambda row: {"actual": "A", "hit": True} if row["label"] == "one" else None)

    state = status(SCHEMA, path)
    assert (state["n_recorded"], state["n_scored"], state["n_pending"]) == (2, 1, 1)
    assert state["next_event"] == pd.Timestamp(TOMORROW)


def test_status_says_nothing_about_how_good_the_predictions_are(path):
    """What counts as resolution is a domain question — the lottery attaches its
    minimum detectable effect, and core/ deliberately attaches nothing."""
    assert "min_detectable_effect" not in status(SCHEMA, path)


def test_pending_needs_either_a_path_or_a_frame():
    """Neither is a caller who has lost track of which, not a request to read
    some default file."""
    with pytest.raises(RegistryError, match="either a path or a registry"):
        pending(SCHEMA)
