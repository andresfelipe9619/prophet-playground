"""The draw store behind the loader — lottery/utils/processor.py's store path.

The store replaces the file, not the contract, and these tests pin exactly that:
a CSV and a store must produce **identical** `(df, balls_expanded)`, a file that
the contract rejects must never reach the store, and a re-import of an
overlapping export must be a no-op rather than a doubling.

The last one is the reason a store exists at all. `final-final.csv` is appended
to by a scraper, and an append that silently duplicates every row of an
overlapping window leaves a file that loads fine, has twice the evidence it
should, and reports every frequency test as far more certain than it is.
"""

import os

import pandas as pd
import pytest

from core.storage import SchemaVersionError, StorageError, read_frame
from lottery.utils.processor import (
    DRAWS_SCHEMA_VERSION,
    DRAWS_TABLE,
    import_csv,
    is_store,
    load_and_preprocess,
    store_status,
)
from lottery.utils.sample_data import load_sample_and_preprocess


@pytest.fixture(scope="module")
def raw_csv(tmp_path_factory):
    """The synthetic draws written back out in the on-disk contract's shape."""
    df, balls = load_sample_and_preprocess()
    path = os.path.join(str(tmp_path_factory.mktemp("draws")), "final-final.csv")
    pd.DataFrame({
        "Date": df["ds"].dt.strftime("%d/%m/%Y"),
        "Ball": balls.astype(int).astype(str).agg("-".join, axis=1),
    }).to_csv(path, index=False)
    return path


def test_a_store_and_a_csv_load_to_the_same_thing(raw_csv, tmp_path):
    store = os.path.join(str(tmp_path), "draws.sqlite")
    import_csv(raw_csv, store)

    from_csv, balls_csv = load_and_preprocess(raw_csv)
    from_store, balls_store = load_and_preprocess(store)

    pd.testing.assert_frame_equal(from_csv, from_store)
    pd.testing.assert_frame_equal(balls_csv, balls_store)


def test_the_store_holds_the_raw_contract_rows(raw_csv, tmp_path):
    # Not the tidy frame: `preprocess_draws` stays the single owner of what a
    # draw looks like, and the store is just where the rows live.
    store = os.path.join(str(tmp_path), "draws.sqlite")
    import_csv(raw_csv, store)
    assert list(read_frame(store, DRAWS_TABLE).columns) == ["Date", "Ball"]
    assert store_status(store)["schema_version"] == DRAWS_SCHEMA_VERSION


def test_a_file_the_contract_rejects_never_reaches_the_store(tmp_path):
    bad = os.path.join(str(tmp_path), "bad.csv")
    pd.DataFrame({"Date": ["01/01/2024"], "Number": ["1-2-3"]}).to_csv(bad, index=False)
    store = os.path.join(str(tmp_path), "draws.sqlite")

    with pytest.raises(Exception, match="(?i)ball|column"):
        import_csv(bad, store)
    # Validated before the write, which is the point of having a store at all.
    assert not os.path.exists(store)


def test_reimporting_the_same_export_is_a_no_op(raw_csv, tmp_path):
    store = os.path.join(str(tmp_path), "draws.sqlite")
    first = import_csv(raw_csv, store)
    again = import_csv(raw_csv, store)
    assert again["n_rows"] == first["n_rows"]
    assert again["fingerprint"] == first["fingerprint"]


def test_a_new_export_that_extends_the_old_one_appends_only_what_is_new(raw_csv, tmp_path):
    store = os.path.join(str(tmp_path), "draws.sqlite")
    published = pd.read_csv(raw_csv)
    earlier = os.path.join(str(tmp_path), "earlier.csv")
    published.head(50).to_csv(earlier, index=False)

    import_csv(earlier, store)
    info = import_csv(raw_csv, store)
    assert info["n_rows"] == len(published)


def test_a_duplicate_is_a_refusal_when_the_caller_has_not_decided(raw_csv, tmp_path):
    store = os.path.join(str(tmp_path), "draws.sqlite")
    import_csv(raw_csv, store)
    with pytest.raises(StorageError, match="already stored"):
        import_csv(raw_csv, store, on_duplicate="error")


def test_a_store_written_under_another_schema_version_is_refused(raw_csv, tmp_path):
    from core.storage import write_frame

    store = os.path.join(str(tmp_path), "draws.sqlite")
    write_frame(pd.read_csv(raw_csv), store, DRAWS_TABLE, "baloto-draws-0")
    with pytest.raises(SchemaVersionError, match="schema version"):
        load_and_preprocess(store)


def test_the_two_kinds_of_path_are_told_apart_by_name(tmp_path):
    assert is_store("exported_data/draws.sqlite")
    assert is_store("/tmp/a.db")
    assert not is_store("exported_data/final-final.csv")


# ------------------------------------------------------- the scheduled-job CLI

def test_check_fails_on_a_source_that_has_gone_quiet(raw_csv, tmp_path, capsys):
    # The failure a schedule exists to catch, and the one a cron job that only
    # logs will hide: the scraper keeps exiting 0 and appending nothing.
    from scripts.store_sync import main

    store = os.path.join(str(tmp_path), "draws.sqlite")
    assert main(["--store", store, "import", "--csv", raw_csv]) == 0
    assert main(["--store", store, "check"]) == 1
    assert "no draw in" in capsys.readouterr().err


def test_check_passes_when_the_store_is_current(tmp_path):
    from scripts.store_sync import main

    today = pd.Timestamp.utcnow().normalize()
    recent = os.path.join(str(tmp_path), "recent.csv")
    pd.DataFrame({
        "Date": [d.strftime("%d/%m/%Y") for d in pd.date_range(today - pd.Timedelta(days=4),
                                                               periods=3, freq="2D")],
        "Ball": ["3-12-19-27-41-8", "1-5-9-33-40-2", "7-11-22-30-43-16"],
    }).to_csv(recent, index=False)

    store = os.path.join(str(tmp_path), "draws.sqlite")
    assert main(["--store", store, "import", "--csv", recent]) == 0
    assert main(["--store", store, "check", "--max-age-days", "7"]) == 0


def test_check_fails_on_a_missing_store(tmp_path, capsys):
    from scripts.store_sync import main

    assert main(["--store", os.path.join(str(tmp_path), "nothing.sqlite"), "check"]) == 1
    assert "no draw table" in capsys.readouterr().err.lower()
