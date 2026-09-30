"""The three domains' job handlers — lottery/jobs.py, football/jobs.py, cycling/jobs.py.

One property matters here and each test pins it for one surface: **a result
read from the job store is the result the dashboard button used to compute
directly.** Moving work out of the request is only an improvement if it changes
nothing about the answer — the inputs have to survive the store (football's
`odds_source` attrs among them), and the handler has to rebuild exactly what the
page would have built.
"""

import pandas as pd
import pytest

import cycling.jobs as cj
import football.jobs as fj
import lottery.backtest as bt
import lottery.jobs as lj
from core import jobs
from cycling.processor import preprocess_results
from cycling.sample_data import generate_stage_race
from football.backtest import compare_models
from football.processor import preprocess_matches
from football.sample_data import generate_matches
from lottery.models.common import build_position_series

# Wall-clock and provenance columns: identical runs differ here by construction.
VOLATILE = {"manifest"}


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "jobs.sqlite")


def _result(store, job):
    assert job["status"] == jobs.DONE, job["error"]
    return jobs.load_result(store, job["id"])


# ------------------------------------------------------------------ lottery

def test_the_draw_input_rebuilds_the_same_position_series(sample):
    """Positions are the thing this codebase is most careful about. Renaming
    the columns to b0..b5 for storage must not move one."""
    df, balls_expanded = sample
    draws = lj.draws_input(df, balls_expanded)

    rebuilt, n_columns = lj._position_series(draws)
    original = build_position_series(df, balls_expanded)

    assert list(draws.columns) == ["ds", *(f"b{p}" for p in range(balls_expanded.shape[1]))]
    assert n_columns == balls_expanded.shape[1]
    for position in original:
        pd.testing.assert_frame_equal(rebuilt[position], original[position])


def test_positions_are_ordered_numerically_not_as_strings():
    """With ten or more columns 'b10' sorts before 'b2' as a string."""
    order = [10, 2, 0, 11, 1, 3, 4, 5, 6, 7, 8, 9]   # deliberately not in position order
    draws = pd.DataFrame({"ds": pd.to_datetime(["2024-01-03"]),
                          **{f"b{p}": [p + 1] for p in order}})

    series, n_columns = lj._position_series(draws)

    assert n_columns == 12
    assert [int(series[p]["y"].iloc[0]) for p in range(12)] == list(range(1, 13))


@pytest.mark.slow
def test_the_queued_walk_forward_is_the_backtest_the_page_ran(store, sample):
    df, balls_expanded = sample
    params = {"n_windows": 3, "min_train": 150}

    job = jobs.run_here(store, lj.WALK_FORWARD, params,
                        {"draws": lj.draws_input(df, balls_expanded)}, lj.walk_forward)
    queued = _result(store, job)["summary"]

    direct = bt.summarize(bt.run_all(build_position_series(df, balls_expanded),
                                     balls_expanded.shape[1], **params))
    pd.testing.assert_frame_equal(queued, direct, check_exact=True)
    assert queued.attrs["manifest"]["inputs"] == direct.attrs["manifest"]["inputs"]


@pytest.mark.slow
def test_the_queued_holdout_carries_its_detail_and_its_info(store, sample):
    df, balls_expanded = sample
    cutoff = str(df["ds"].iloc[-5].date())

    job = jobs.run_here(store, lj.HOLDOUT, {"cutoff": cutoff, "mode": "frozen"},
                        {"draws": lj.draws_input(df, balls_expanded)}, lj.holdout)
    result = _result(store, job)

    assert result["info"]["n_holdout"] == 4
    assert result["info"]["cutoff"] == pd.Timestamp(cutoff)
    assert len(result["detail"]) == 4
    assert set(result["summary"]["model"]) >= {"FrequencyBaseline", "XGBoost"}


# ------------------------------------------------------------------ football

@pytest.fixture
def matches():
    raw = generate_matches(n_teams=8, seed=1).drop(columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


FOOTBALL_PARAMS = {"n_windows": 6, "min_train": 40, "half_life": None,
                   "method": "multiplicative", "models": ["elo", "blend"],
                   "blend_weight": 0.5, "pool": "linear"}


def test_the_match_frame_reaches_the_worker_with_its_odds_source(store, matches):
    """Without `odds_source` the worker would not know which market it is
    holding the model to — the attrs are part of the input, not decoration."""
    seen = {}

    def spy(params, inputs, progress):
        seen.update(inputs["matches"].attrs)
        return {"ok": True}

    jobs.run_here(store, fj.COMPARE_MODELS, FOOTBALL_PARAMS, {"matches": matches}, spy)

    assert seen == matches.attrs
    assert seen["odds_are_closing"] is True


def test_the_queued_market_test_is_the_one_the_page_ran(store, matches):
    job = jobs.run_here(store, fj.COMPARE_MODELS, FOOTBALL_PARAMS, {"matches": matches},
                        fj.run_compare_models)
    queued = _result(store, job)["table"]

    direct = compare_models(matches, n_windows=6, min_train=40, models=("elo", "blend"))
    pd.testing.assert_frame_equal(queued, direct, check_exact=True)
    # The held-out forecasts the calibration chart reads come back as arrays.
    for name, matrix in direct.attrs["forecasts"]["models"].items():
        assert (queued.attrs["forecasts"]["models"][name] == matrix).all()


def test_a_half_life_of_zero_means_no_decay_as_it_does_on_the_page(store, matches):
    seen = {}

    def spy(params, inputs, progress):
        seen["table"] = fj.run_compare_models({**params, "models": ["elo"]}, inputs, progress)["table"]
        return {"ok": True}

    jobs.run_here(store, fj.COMPARE_MODELS, {**FOOTBALL_PARAMS, "half_life": 0},
                  {"matches": matches}, spy)

    assert seen["table"]["half_life"].tolist() == [None]


# ------------------------------------------------------------------ cycling

def test_the_queued_walk_forward_against_the_ranking_is_the_one_the_page_ran(store):
    results = preprocess_results(generate_stage_race(seed=0, n_riders=20, n_stages=6),
                                 validate=False)
    params = {"metric": "plackett_luce", "min_history": 2}

    job = jobs.run_here(store, cj.COMPARE_FORECASTERS, params, {"results": results},
                        cj.run_compare_forecasters)
    queued = _result(store, job)

    table, scores = cj.run_compare_forecasters(params, {"results": results}, lambda _: None).values()
    pd.testing.assert_frame_equal(queued["table"], table, check_exact=True)
    pd.testing.assert_frame_equal(queued["scores"], scores, check_exact=True)


def test_the_uniform_draw_is_always_in_the_cycling_comparison():
    """It loses to the ranking, which is the demonstration that it is not a
    baseline — so the job has no way to leave it out."""
    assert set(cj.FORECASTERS) == {"ranking", "plackett_luce", "uniform"}


def test_every_domain_names_its_jobs_under_its_own_prefix():
    for prefix, handlers in (("lottery.", lj.HANDLERS), ("football.", fj.HANDLERS),
                             ("cycling.", cj.HANDLERS)):
        assert handlers and all(kind.startswith(prefix) for kind in handlers)
