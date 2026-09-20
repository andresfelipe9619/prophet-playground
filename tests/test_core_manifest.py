"""What a manifest must record, and what a fingerprint must notice.

The invariants here are all about *failing to distinguish* — a manifest that
reports a commit while the tree was edited, or a fingerprint that hashes two
different frames to the same digest, is worse than none at all, because it
reads as provenance while providing none.
"""

import json

import numpy as np
import pandas as pd
import pytest

from core.manifest import (
    NUMERIC_LIBRARIES,
    data_fingerprint,
    git_revision,
    library_versions,
    run_manifest,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-06"]),
            "team": ["A", "B", "C"],
            "value": [1.5, 2.5, 3.5],
            "n": [1, 2, 3],
        }
    )


# --------------------------------------------------------------- run_manifest

def test_a_manifest_is_json_serialisable():
    """It is written beside a result, not only printed, so it has to serialise."""
    text = json.dumps(run_manifest({"data": "abc", "seed": 0}))
    assert json.loads(text)["inputs"] == {"data": "abc", "seed": 0}


def test_a_manifest_records_the_inputs_it_was_given_without_interpreting_them():
    manifest = run_manifest({"n_windows": 20, "cutoff": "2026-07-31", "seeds": [1, 2]})
    assert manifest["inputs"] == {"n_windows": 20, "cutoff": "2026-07-31", "seeds": [1, 2]}


def test_a_manifest_with_no_inputs_still_has_the_key():
    """An empty dict, never a missing key: absent reads as an oversight."""
    assert run_manifest()["inputs"] == {}


def test_a_manifest_names_every_library_that_can_move_a_number():
    libraries = run_manifest()["libraries"]
    assert set(libraries) == set(NUMERIC_LIBRARIES)


def test_an_absent_library_is_recorded_as_none_rather_than_dropped():
    versions = library_versions(("numpy", "a-package-that-is-not-installed"))
    assert versions["numpy"] is not None
    assert versions["a-package-that-is-not-installed"] is None


def test_the_git_revision_has_all_three_fields_whether_or_not_git_answered():
    """Called from a checkout or from a tarball, the shape does not change."""
    assert set(git_revision()) == {"commit", "dirty", "branch"}


def test_the_manifest_carries_the_git_block():
    assert set(run_manifest()["git"]) == {"commit", "dirty", "branch"}


# ----------------------------------------------------------- data_fingerprint

def test_the_same_frame_fingerprints_the_same_twice(frame):
    assert data_fingerprint(frame) == data_fingerprint(frame.copy())


def test_one_edited_cell_changes_the_fingerprint(frame):
    """The failure mode this exists for: row counts and date ranges miss it."""
    edited = frame.copy()
    edited.loc[1, "value"] = 2.6
    assert len(edited) == len(frame) and edited["ds"].equals(frame["ds"])
    assert data_fingerprint(edited) != data_fingerprint(frame)


def test_one_edited_label_changes_the_fingerprint(frame):
    edited = frame.copy()
    edited.loc[2, "team"] = "D"
    assert data_fingerprint(edited) != data_fingerprint(frame)


def test_reordered_columns_change_the_fingerprint(frame):
    """Position semantics: the same values in a different column order are not
    the same data to anything downstream, so they are not the same fingerprint."""
    reordered = frame[["team", "ds", "n", "value"]]
    assert set(reordered.columns) == set(frame.columns)
    assert data_fingerprint(reordered) != data_fingerprint(frame)


def test_a_renamed_column_changes_the_fingerprint(frame):
    renamed = frame.rename(columns={"value": "score"})
    assert data_fingerprint(renamed) != data_fingerprint(frame)


def test_a_changed_dtype_changes_the_fingerprint(frame):
    """int and float columns of the same numbers behave differently downstream."""
    as_float = frame.copy()
    as_float["n"] = as_float["n"].astype(float)
    assert data_fingerprint(as_float) != data_fingerprint(frame)


def test_reordered_rows_change_the_fingerprint(frame):
    """Every domain here is time-ordered; a shuffled frame is a different input."""
    shuffled = frame.iloc[[2, 0, 1]]
    assert data_fingerprint(shuffled) != data_fingerprint(frame)


def test_a_sliced_frame_fingerprints_without_raising(frame):
    """A slice can hand over a non-contiguous buffer, which `tobytes` refuses."""
    assert data_fingerprint(frame.iloc[1:]) != data_fingerprint(frame)


def test_the_fingerprint_survives_a_round_trip_through_csv(tmp_path):
    """The claim being pinned is narrow: re-reading the same file gives the same
    digest. It is not a claim that any two representations of the same numbers
    agree — a float written and re-parsed is the case that would break that."""
    original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "frame.csv"
    original.to_csv(path, index=False)
    first = pd.read_csv(path)
    second = pd.read_csv(path)
    assert data_fingerprint(first) == data_fingerprint(second)


def test_an_empty_frame_fingerprints(frame):
    empty = frame.iloc[:0]
    assert data_fingerprint(empty) != data_fingerprint(frame)
    assert data_fingerprint(empty) == data_fingerprint(frame.iloc[:0].copy())


def test_nan_does_not_make_a_frame_unfingerprintable():
    """NaN != NaN, so a digest built on comparison would be unstable here."""
    with_nan = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert data_fingerprint(with_nan) == data_fingerprint(with_nan.copy())


# ------------------------------- every evaluation entry point attaches one
#
# The module above can be perfect and still useless if no result carries a
# manifest. These pin the wiring, which is the part that rots: a new backtest
# entry point added without one looks identical in every results table.

def _manifest_shape(manifest):
    assert set(manifest) == {"generated_at", "git", "python", "platform", "libraries", "inputs"}
    assert "data" in manifest["inputs"], "a manifest with no data fingerprint names no data"
    json.dumps(manifest)


@pytest.mark.slow
def test_the_lottery_walk_forward_attaches_a_manifest():
    from lottery import backtest as bt
    from lottery.models.common import build_position_series
    from lottery.utils.sample_data import load_sample_and_preprocess

    df, balls = load_sample_and_preprocess(n_draws=90)
    series = build_position_series(df, balls)
    results = bt.run_all(series, balls.shape[1], n_windows=3, min_train=60)
    for frame in results.values():
        _manifest_shape(frame.attrs["manifest"])
    # The summary is the table anyone actually reads, so it is the one that
    # most needs to be able to say what produced it.
    _manifest_shape(bt.summarize(results).attrs["manifest"])


@pytest.mark.slow
def test_the_football_comparison_attaches_a_manifest():
    from football.backtest import compare_models
    from football.processor import preprocess_matches
    from football.sample_data import generate_matches

    raw = generate_matches(n_teams=10, seed=3, market_noise=0.3).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    matches = preprocess_matches(raw, validate=False)
    table = compare_models(matches, n_windows=4, min_train=60, models=("elo",))
    _manifest_shape(table.attrs["manifest"])
    # The odds source belongs in the manifest for the same reason the two eras
    # of Baloto do: it changes what the result means and not what it looks like.
    assert "odds_are_closing" in table.attrs["manifest"]["inputs"]


def test_the_cycling_walk_forward_attaches_a_manifest():
    from cycling.baseline import uniform_worths
    from cycling.evaluation import compare_forecasters, walk_forward
    from cycling.processor import preprocess_results
    from cycling.sample_data import generate_stage_race

    results = preprocess_results(
        generate_stage_race(n_riders=20, n_stages=4, seed=1), validate=False)
    forecasters = {
        "uniform": lambda history, riders, as_of: uniform_worths(len(riders)),
        "also_uniform": lambda history, riders, as_of: uniform_worths(len(riders)),
    }
    scores = walk_forward(results, forecasters)
    _manifest_shape(scores.attrs["manifest"])

    table, _ = compare_forecasters(results, forecasters, baseline="uniform")
    _manifest_shape(table.attrs["manifest"])
    assert table.attrs["manifest"]["inputs"]["baseline"] == "uniform"
