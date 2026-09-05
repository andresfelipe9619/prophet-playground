"""Walk-forward evaluation — lottery/backtest.py.

Two invariants carry the file: scoring is set-based (which makes the exact
hypergeometric baseline the right comparison, and makes order-statistic
artifacts in the source data unable to inflate results), and the summary
reports a multiplicity-corrected verdict alongside the naive one.
"""

import pandas as pd
import pytest

import lottery.backtest as bt
from lottery.models.common import build_position_series


# -------------------------------------------------------------------- splits

@pytest.mark.parametrize(
    "n_draws, n_windows, min_train, expected",
    [
        (200, 15, 60, (185, 200)),   # the last 15 draws
        (200, 500, 60, (60, 200)),   # more windows than history: min_train wins
        (100, 15, 100, (100, 100)),  # nothing left to evaluate
    ],
)
def test_window_bounds(n_draws, n_windows, min_train, expected):
    assert bt.window_bounds(n_draws, n_windows, min_train) == expected


def test_cutoff_counts_the_cutoff_day_as_training():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-06", "2024-01-08"])
    assert bt.cutoff_bounds(dates, "2024-01-03") == (2, 2)


def test_cutoff_before_the_history_trains_on_nothing():
    dates = pd.to_datetime(["2024-01-03", "2024-01-06"])
    assert bt.cutoff_bounds(dates, "2023-12-31") == (0, 2)


def test_cutoff_after_the_history_holds_out_nothing():
    dates = pd.to_datetime(["2024-01-03", "2024-01-06"])
    assert bt.cutoff_bounds(dates, "2024-06-01") == (2, 0)


def test_cutoff_is_order_independent():
    """Callers may pass an unsorted date column."""
    shuffled = pd.to_datetime(["2024-01-08", "2024-01-01", "2024-01-06", "2024-01-03"])
    assert bt.cutoff_bounds(shuffled, "2024-01-03") == (2, 2)


# ------------------------------------------------------------------- scoring

ACTUAL = {0: 3, 1: 12, 2: 19, 3: 27, 4: 41, 5: 8}


def test_scoring_is_order_agnostic():
    """The same five numbers in different slots must score identically."""
    in_order = {0: 3, 1: 12, 2: 19, 3: 27, 4: 41, 5: 8}
    shuffled = {0: 41, 1: 27, 2: 19, 3: 12, 4: 3, 5: 8}
    assert bt._score_window(in_order, ACTUAL, 6) == bt._score_window(shuffled, ACTUAL, 6) == (5, True, 5)


def test_scoring_counts_the_intersection():
    preds = {0: 3, 1: 12, 2: 40, 3: 42, 4: 43, 5: 1}
    main_hits, super_hit, m_guessed = bt._score_window(preds, ACTUAL, 6)
    assert (main_hits, super_hit, m_guessed) == (2, False, 5)


def test_m_guessed_counts_distinct_numbers_not_slots():
    """Collisions shrink the guess set, which changes the chance baseline the
    z-test compares against."""
    preds = {0: 3, 1: 3, 2: 3, 3: 27, 4: 41, 5: 8}
    main_hits, _, m_guessed = bt._score_window(preds, ACTUAL, 6)
    assert m_guessed == 3
    assert main_hits == 3


def test_a_repeated_guess_cannot_be_counted_twice():
    preds = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 8}
    assert bt._score_window(preds, ACTUAL, 6) == (1, True, 1)


def test_the_superbalota_is_scored_separately():
    """It is not part of the main set — matching it must not add a main hit."""
    preds = {0: 40, 1: 42, 2: 43, 3: 44 - 1, 4: 39, 5: 8}
    main_hits, super_hit, _ = bt._score_window(preds, ACTUAL, 6)
    assert main_hits == 0 and super_hit is True


def test_scoring_does_not_assume_six_columns():
    """Position semantics come from the helpers, so a different width still works."""
    actual = {0: 3, 1: 12, 2: 19, 3: 8}
    preds = {0: 19, 1: 12, 2: 3, 3: 8}
    assert bt._score_window(preds, actual, 4) == (3, True, 3)


# ------------------------------------------------------------------- summary

def _results(hits, m_guessed=5):
    return pd.DataFrame(
        [{"t": i, "main_hits": h, "super_hit": False, "m_guessed": m_guessed}
         for i, h in enumerate(hits)],
        columns=bt.RESULT_COLUMNS,
    )


def test_summarize_reports_both_verdicts():
    summary = bt.summarize({"A": _results([0, 1, 0, 1]), "B": _results([0, 0, 1, 0])})
    assert {"beats_chance", "beats_chance_corrected", "bonferroni_threshold",
            "p_value_better_than_chance", "chance_avg_main_hits"} <= set(summary.columns)
    assert (summary["bonferroni_threshold"] == 0.05 / 2).all()


def test_the_bonferroni_threshold_tightens_with_more_models():
    six = {name: _results([0, 1, 0]) for name in "ABCDEF"}
    assert (bt.summarize(six)["bonferroni_threshold"] == 0.05 / 6).all()


def test_the_corrected_verdict_is_never_more_permissive():
    models = {"lucky": _results([2, 2, 2, 2, 2, 2, 2, 2]), "plain": _results([0, 0, 1, 0])}
    summary = bt.summarize(models)
    assert not (summary["beats_chance_corrected"] & ~summary["beats_chance"]).any()


def test_summarize_is_ordered_best_first():
    summary = bt.summarize({"weak": _results([0, 0, 0]), "strong": _results([3, 3, 3])})
    assert list(summary["model"]) == ["strong", "weak"]


def test_summarize_compares_against_the_chance_mean():
    summary = bt.summarize({"A": _results([0, 1, 0, 1])})
    assert summary["chance_avg_main_hits"].iloc[0] == pytest.approx(5 * 5 / 43)
    assert summary["chance_super_hit_rate"].iloc[0] == pytest.approx(1 / 16)


# ---------------------------------------------------------- the loop itself

def test_a_window_without_a_prediction_is_skipped_not_scored(position_series):
    """Scoring a missing prediction would hand the model free hits."""

    @bt._predictor("Sometimes")
    def predict(series, t):
        return {"Sometimes": None if t % 2 else bt.most_frequent_pick(series, upto=t)}

    results = bt._run_windows(position_series, 6, n_windows=10, min_train=60,
                              predict_window=predict)
    assert 0 < len(results["Sometimes"]) < 10


def test_the_loop_refuses_an_impossible_split(position_series):
    with pytest.raises(ValueError, match="No windows to evaluate"):
        bt._run_windows(position_series, 6, n_windows=10,
                        min_train=len(position_series[0]) + 5,
                        predict_window=bt._frequency_window)


def test_the_frequency_baseline_never_sees_the_future(position_series):
    """Its pick at t must equal the pick computed from the truncated history alone."""
    preds = bt._frequency_window(position_series, 100)["FrequencyBaseline"]
    truncated = {p: f.iloc[:100] for p, f in position_series.items()}
    assert preds == bt.most_frequent_pick(truncated)


def test_run_holdout_rejects_an_unknown_mode(position_series):
    with pytest.raises(ValueError, match="mode must be one of"):
        bt.run_holdout(position_series, 6, "2018-06-01", mode="magic")


def test_run_holdout_refuses_a_cutoff_with_nothing_after_it(position_series):
    last = position_series[0]["ds"].max()
    with pytest.raises(ValueError, match="nothing to predict"):
        bt.run_holdout(position_series, 6, last)


def test_run_holdout_refuses_a_cutoff_with_too_little_training(position_series):
    first = position_series[0]["ds"].min()
    with pytest.raises(ValueError, match="need at least|at least"):
        bt.run_holdout(position_series, 6, first)


@pytest.mark.slow
def test_run_all_end_to_end(position_series):
    """Every model runs and produces a scoreable, in-range result.

    Note what is deliberately *not* asserted: that no model beats chance. On a
    handful of windows the z-test is dominated by noise and a signal-free model
    clears the bar often enough to make such an assertion flaky — which is the
    project's own thesis, not a defect. What is asserted instead is that the
    observed hit rate stays in the neighbourhood of the chance rate, and that
    the corrected verdict never contradicts the naive one.
    """
    results = bt.run_all(position_series, 6, n_windows=25, min_train=100)
    assert set(results) >= {"FrequencyBaseline", "XGBoost", "AutoARIMA", "AutoETS", "AutoTheta"}
    for name, frame in results.items():
        assert list(frame.columns) == bt.RESULT_COLUMNS
        assert frame["main_hits"].between(0, 5).all()
        assert frame["m_guessed"].between(1, 5).all()
        assert len(frame) == 25

    summary = bt.summarize(results)
    assert (summary["avg_main_hits"] - summary["chance_avg_main_hits"]).abs().max() < 1.0
    assert not (summary["beats_chance_corrected"] & ~summary["beats_chance"]).any()


@pytest.mark.slow
def test_holdout_detail_lists_every_drawn_number(position_series, sample):
    df, balls = sample
    cutoff = df["ds"].iloc[-4]
    results, info = bt.run_holdout(position_series, 6, cutoff, mode="frozen")
    assert info["n_holdout"] == 3
    assert info["mode"] == "frozen"

    detail = bt.holdout_detail(results, position_series, 6)
    assert len(detail) == 3
    assert list(detail.columns[:3]) == ["ds", "sorteo", "superbalota"]
    assert detail["sorteo"].iloc[0].count("-") == 4
