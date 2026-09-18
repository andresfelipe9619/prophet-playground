"""TimesFM's invariants, pinned without the 200M-parameter checkpoint.

Every assertion here is about **this repository's** contract with the model, not
about the model's forecasting quality: that the draw being predicted never
reaches the context, that raw output is clipped into each position's legal range,
that all positions ride in one batched call, and that a short history is refused
rather than guessed at. Those are the properties a refactor breaks silently, and
none of them needs real weights to check.

The seam is `forecaster=`, which takes anything exposing
`.forecast(horizon, inputs)`. The stub below records what it was handed, so the
tests can assert on the *context* — the one thing a temporal leak would show up
in and the one thing a numeric comparison against real weights would not reveal.

This is also why the module is testable at all in CI: `requirements-test.txt`
installs neither timesfm nor torch, and this file imports neither.
"""

import numpy as np
import pandas as pd
import pytest

from lottery.models.common import MAIN_BALL_RANGE, SUPER_BALL_RANGE, build_position_series
from lottery.models.timesfm_model import (
    MIN_CONTEXT,
    forecast_horizon_positions,
    forecast_positions,
    is_available,
)
from lottery.utils.sample_data import load_sample_and_preprocess

N_COLUMNS = 6


class StubForecaster:
    """Records its inputs and returns a constant, so the assertions are about us.

    `value` is deliberately out of range for every position — a model that knew
    nothing about Baloto would return exactly this kind of number, and clipping
    is what stands between it and a ticket that cannot exist.
    """

    def __init__(self, value=999.0):
        self.value = value
        self.calls = []

    def forecast(self, horizon, inputs):
        self.calls.append({"horizon": horizon, "inputs": [np.asarray(a) for a in inputs]})
        point = np.full((len(inputs), horizon), self.value, dtype=np.float32)
        quantiles = np.zeros((len(inputs), horizon, 9), dtype=np.float32)
        return point, quantiles


@pytest.fixture
def series():
    df, balls = load_sample_and_preprocess(n_draws=200)
    return build_position_series(df, balls)


def test_availability_check_does_not_import_torch(monkeypatch):
    # The dashboard calls this on every rerun. If it imported torch to answer,
    # it would cost seconds and gigabytes on a surface Streamlit re-executes top
    # to bottom — which is the whole reason the check uses find_spec.
    import sys

    before = set(sys.modules)
    is_available()
    newly_imported = {m for m in set(sys.modules) - before if m.split(".")[0] in ("torch", "timesfm")}
    assert not newly_imported, f"is_available() imported {newly_imported}"


def test_every_position_rides_in_one_batched_call(series):
    stub = StubForecaster()
    forecast_positions(series, N_COLUMNS, forecaster=stub)

    assert len(stub.calls) == 1, "one forward pass should cover every position"
    assert len(stub.calls[0]["inputs"]) == N_COLUMNS


def test_output_is_clipped_into_each_positions_range(series):
    # The stub returns 999 for everything. TimesFM knows nothing about 1-43 or
    # 1-16 and would just as happily return 44.7 or a negative number.
    preds = forecast_positions(series, N_COLUMNS, forecaster=StubForecaster(999.0))

    for position in range(N_COLUMNS - 1):
        assert preds[position] == MAIN_BALL_RANGE[1]
    assert preds[N_COLUMNS - 1] == SUPER_BALL_RANGE[1]

    preds_low = forecast_positions(series, N_COLUMNS, forecaster=StubForecaster(-5.0))
    assert set(preds_low.values()) == {MAIN_BALL_RANGE[0]}, "a negative forecast clips to the floor"


def test_the_predicted_draw_is_never_in_the_context(series):
    # The leak this whole project is shaped against. `upto=t` must cut the
    # context strictly before t, so the value being predicted cannot inform it.
    t = 150
    stub = StubForecaster()
    forecast_positions(series, N_COLUMNS, upto=t, forecaster=stub)

    for position, context in enumerate(stub.calls[0]["inputs"]):
        expected = series[position]["y"].to_numpy()[:t]
        assert len(context) == len(expected[-len(context):])
        np.testing.assert_array_equal(context, expected[-len(context):])
        # The decisive assertion: the value at t is absent from the tail.
        assert context[-1] == pytest.approx(series[position]["y"].iloc[t - 1])


def test_context_is_truncated_to_max_context(series):
    stub = StubForecaster()
    forecast_positions(series, N_COLUMNS, forecaster=stub, max_context=64)

    for context in stub.calls[0]["inputs"]:
        assert len(context) == 64, "a longer history must be cut to the compiled context"


def test_a_short_history_is_refused_rather_than_guessed(series):
    short = {pos: frame.iloc[: MIN_CONTEXT - 1] for pos, frame in series.items()}
    stub = StubForecaster()

    assert forecast_positions(short, N_COLUMNS, forecaster=stub) is None
    assert not stub.calls, "the model should not even be called without enough context"


def test_horizon_mode_returns_one_map_per_step(series):
    stub = StubForecaster()
    by_step = forecast_horizon_positions(series, N_COLUMNS, start=100, horizon=5, forecaster=stub)

    assert sorted(by_step) == [0, 1, 2, 3, 4]
    assert all(sorted(step) == list(range(N_COLUMNS)) for step in by_step.values())
    assert stub.calls[0]["horizon"] == 5
    # Frozen means frozen: nothing at or after `start` may reach the context.
    for position, context in enumerate(stub.calls[0]["inputs"]):
        assert context[-1] == pytest.approx(series[position]["y"].iloc[99])


def test_it_reaches_the_backtest_and_is_corrected_with_the_others(series, monkeypatch):
    """The wiring, end to end, with the checkpoint stubbed out.

    This is the assertion the module-level tests cannot make: that a run asking
    for TimesFM actually scores it, and that adding it **tightens** the
    Bonferroni threshold for every model rather than quietly getting a private
    uncorrected verdict. A foundation model is one more chance at a false
    positive, not an exception to the correction.
    """
    import lottery.backtest as bt
    import lottery.models.timesfm_model as tfm

    monkeypatch.setattr(tfm, "load_forecaster", lambda **kwargs: StubForecaster(7.0))

    without = bt.summarize(bt.run_all(series, N_COLUMNS, n_windows=3, min_train=100))
    with_tfm = bt.summarize(
        bt.run_all(series, N_COLUMNS, n_windows=3, min_train=100, include_timesfm=True))

    assert "TimesFM" not in set(without["model"])
    assert "TimesFM" in set(with_tfm["model"])
    assert len(with_tfm) == len(without) + 1
    # One more model tested against the same draws means a stricter bar for all.
    assert with_tfm["bonferroni_threshold"].iloc[0] < without["bonferroni_threshold"].iloc[0]
    # And it is held to the same verdict columns as everything else.
    assert {"beats_chance", "beats_chance_corrected"} <= set(with_tfm.columns)


def test_frozen_holdout_includes_it(series, monkeypatch):
    import lottery.backtest as bt
    import lottery.models.timesfm_model as tfm

    monkeypatch.setattr(tfm, "load_forecaster", lambda **kwargs: StubForecaster(7.0))

    cutoff = series[0]["ds"].iloc[150]
    results, info = bt.run_holdout(series, N_COLUMNS, cutoff, mode="frozen", include_timesfm=True)

    assert "TimesFM" in results
    assert len(results["TimesFM"]) == info["n_holdout"]


class _Fake3Output:
    def __init__(self, values):
        self.forecast = np.asarray(values, dtype=np.float32)


class Fake3Forecaster:
    """Mimics TimesFM 3.0's `predict_batch`, which is a different API from 2.5's."""

    def __init__(self, value=7.0):
        self.value = value
        self.calls = []

    def predict_batch(self, contexts, horizon, **kwargs):
        self.calls.append({"contexts": list(contexts), "horizon": horizon})
        # An iterator of per-series objects, each carrying a (horizon,) forecast.
        return iter([_Fake3Output([self.value] * horizon) for _ in contexts])


def test_the_3_0_adapter_matches_the_2_5_contract():
    # 2.5 returns (batch, horizon) from `.forecast`; 3.0 yields one object per
    # series from `.predict_batch`. The adapter is what stops that difference
    # reaching forecast_matrix, the backtest, the dashboard and every test here.
    from lottery.models.timesfm_model import _Timesfm3Adapter

    adapter = _Timesfm3Adapter(Fake3Forecaster(7.0))
    point, _quantiles = adapter.forecast(horizon=3, inputs=[np.zeros(64), np.zeros(64)])

    assert point.shape == (2, 3)
    assert (point == 7.0).all()


def test_the_3_0_adapter_drives_a_real_forecast(series):
    # End to end through the adapter rather than the stub: the same clipping and
    # position semantics must hold whichever checkpoint family is loaded.
    from lottery.models.timesfm_model import _Timesfm3Adapter

    preds = forecast_positions(series, N_COLUMNS,
                               forecaster=_Timesfm3Adapter(Fake3Forecaster(999.0)))
    for position in range(N_COLUMNS - 1):
        assert preds[position] == MAIN_BALL_RANGE[1]
    assert preds[N_COLUMNS - 1] == SUPER_BALL_RANGE[1]


def test_the_default_context_covers_a_full_history():
    # A real 2010-2026 export is ~1035 draws. The context was cut to 512 while
    # this project was aimed at a hosted free tier; locally there is no reason to
    # hand the model less history than exists.
    from lottery.models.timesfm_model import MAX_CONTEXT

    assert MAX_CONTEXT >= 1035, "the default context should not truncate a full Baloto history"


def test_the_default_checkpoint_is_the_strong_one():
    # Local, non-commercial use, so 3.0 is permitted and is the better model.
    # CHECKPOINT_APACHE stays available for anything published or commercial.
    from lottery.models.timesfm_model import CHECKPOINT, CHECKPOINT_APACHE

    assert "3.0" in CHECKPOINT
    assert "2.5" in CHECKPOINT_APACHE


def test_predictions_are_plain_ints(series):
    # They land in a set with the actual draw in `_score_window`; numpy scalars
    # compare equal but a DataFrame built from them dtypes differently, and the
    # registry writes them to CSV.
    preds = forecast_positions(series, N_COLUMNS, forecaster=StubForecaster(7.4))
    assert all(type(value) is int for value in preds.values())
