"""A store that refuses to lose the shape of what was put in it — core/storage.py.

Every test here is about a failure that a CSV accepts silently.

**Dtype round-trip.** SQLite has no datetime type, so a date column comes back
as text unless something puts it back — and an ISO date string compares
*correctly* against another ISO date string, which is why this bug survives
casual testing and then fails on the first arithmetic or the first non-ISO
value. `test_dates_come_back_as_dates` is the one that would catch it.

**Shape drift on append.** pandas concatenates a float column onto an int one
without complaint, producing a column that means two things. The store refuses
it, which is the domain contracts' load-time guard arriving from the other side.

**An empty write.** A scraper that writes nothing when the markup changes is the
failure mode this project's parsers are shaped against, so the store will not
accept the empty frame that would move that failure one step further away.
"""

import os

import pandas as pd
import pytest

from core.storage import (
    METADATA_TABLE,
    SchemaVersionError,
    StorageError,
    append_frame,
    read_frame,
    table_info,
    tables,
    write_frame,
)

VERSION = "draws-1"


def frame(n=3, start="2024-01-01"):
    return pd.DataFrame({
        "ds": pd.date_range(start, periods=n, freq="D"),
        "label": [f"row {i}" for i in range(n)],
        "value": [float(i) for i in range(n)],
        "count": list(range(n)),
    })


@pytest.fixture
def store(tmp_path):
    return os.path.join(str(tmp_path), "nested", "store.sqlite")


def test_a_frame_round_trips_with_its_metadata(store):
    info = write_frame(frame(), store, "draws", VERSION)
    assert info["schema_version"] == VERSION
    assert info["n_rows"] == 3
    assert info["columns"] == ["ds", "label", "value", "count"]

    back = read_frame(store, "draws", expected_version=VERSION)
    pd.testing.assert_frame_equal(back, frame())
    # The metadata table is the store describing itself and is not a stored table.
    assert tables(store) == ["draws"]
    assert METADATA_TABLE not in tables(store)


def test_dates_come_back_as_dates(store):
    # The whole reason dtypes are recorded and replayed. Text dates sort and
    # compare correctly while they are ISO, so nothing downstream notices until
    # something does arithmetic on them.
    write_frame(frame(), store, "draws", VERSION)
    back = read_frame(store, "draws")
    assert pd.api.types.is_datetime64_any_dtype(back["ds"])
    assert (back["ds"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert back["count"].dtype == frame()["count"].dtype


def test_a_different_schema_version_is_refused(store):
    write_frame(frame(), store, "draws", VERSION)
    with pytest.raises(SchemaVersionError, match="schema version"):
        read_frame(store, "draws", expected_version="draws-2")
    # Left out, anything reads — right for a tool that only inspects a store.
    assert len(read_frame(store, "draws")) == 3


def test_an_empty_frame_is_refused(store):
    with pytest.raises(StorageError, match="empty frame"):
        write_frame(frame().iloc[0:0], store, "draws", VERSION)


def test_appending_a_reordered_or_renamed_column_is_refused(store):
    write_frame(frame(), store, "draws", VERSION)
    reordered = frame(start="2024-02-01")[["label", "ds", "value", "count"]]
    with pytest.raises(StorageError, match="do not match"):
        append_frame(reordered, store, "draws", VERSION)


def test_appending_a_drifted_dtype_is_refused(store):
    # The silent one: pandas would concatenate these and leave a column that
    # loads fine and means something different.
    write_frame(frame(), store, "draws", VERSION)
    drifted = frame(start="2024-02-01")
    drifted["count"] = drifted["count"].astype(float)
    with pytest.raises(StorageError, match="Dtype drift"):
        append_frame(drifted, store, "draws", VERSION)


def test_appending_grows_the_table_and_updates_the_fingerprint(store):
    first = write_frame(frame(), store, "draws", VERSION)
    info = append_frame(frame(start="2024-02-01"), store, "draws", VERSION, key=["ds"])
    assert info["n_rows"] == 6
    # The fingerprint is over values, columns and dtypes, so growth moves it —
    # a row count alone would not distinguish six rows from six other rows.
    assert info["fingerprint"] != first["fingerprint"]
    assert len(read_frame(store, "draws")) == 6


def test_a_rescrape_of_the_same_window_is_a_decision_not_a_default(store):
    write_frame(frame(), store, "draws", VERSION)
    overlapping = frame(n=5)  # the same three days plus two new ones

    with pytest.raises(StorageError, match="already stored"):
        append_frame(overlapping, store, "draws", VERSION, key=["ds"])

    info = append_frame(overlapping, store, "draws", VERSION, key=["ds"], on_duplicate="skip")
    assert info["n_rows"] == 5
    assert read_frame(store, "draws")["ds"].is_monotonic_increasing


def test_a_rescrape_with_nothing_new_leaves_the_store_untouched(store):
    first = write_frame(frame(), store, "draws", VERSION)
    info = append_frame(frame(), store, "draws", VERSION, key=["ds"], on_duplicate="skip")
    # Nothing new is an ordinary outcome for a scheduled scrape and must not
    # look like a failure — but nothing is written either.
    assert info["n_rows"] == 3
    assert info["fingerprint"] == first["fingerprint"]


def test_appending_to_a_table_that_does_not_exist_yet_creates_it(store):
    info = append_frame(frame(), store, "draws", VERSION, key=["ds"])
    assert info["n_rows"] == 3


def test_a_missing_store_or_table_says_so(store):
    with pytest.raises(FileNotFoundError):
        read_frame(store, "draws")
    write_frame(frame(), store, "draws", VERSION)
    with pytest.raises(StorageError, match="No table"):
        read_frame(store, "races")
    assert table_info(store, "races") is None
