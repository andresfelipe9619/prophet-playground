"""Walk-forward backtest: does any model actually beat pure chance?

For each of the last `n_windows` real draws, every model is trained only on
data available before that draw (expanding window, 1-step ahead) and scored
on how many of the 5 main balls it matched and whether it matched the
superbalota. Those hit counts are then compared against the exact
hypergeometric chance baseline (models/baseline.py) with a z-test — this is
the number that should decide whether a model is worth trusting, since raw
"N matches" means nothing without knowing what pure luck would already give
you.

Two ways to choose which draws are held out:

    python backtest.py --n-windows 20 --min-train 100
    python backtest.py --cutoff 2026-07-31 --mode frozen --current-format-only

The first holds out the last N draws. The second holds out everything after a
date — "train on the data up to July, then predict the draws that have already
happened since" — which produces a concrete, checkable result rather than an
average. See `run_holdout` for what the two modes mean.

Prophet is included but off by default (`--include-prophet`) because it
refits per position per window and is far slower than the other models.
"""

import argparse
import time

import pandas as pd

from models.baseline import beats_chance_test, expected_super_match_rate, most_frequent_pick
from models.common import DEFAULT_DATA_PATH, build_position_series, main_positions, super_position
from models.statsforecast_model import MODEL_NAMES, adjusted_predictions, fit_predict_all
from models.xgboost_model import forecast_horizon, train_predict_one_step
from utils.processor import load_and_preprocess

RESULT_COLUMNS = ["t", "main_hits", "super_hit", "m_guessed"]
MIN_TRAIN_FOR_HOLDOUT = 30  # below this the models have nothing to fit


def window_bounds(n_draws, n_windows, min_train):
    """The (start, total) range of draws a backtest would evaluate.

    Single owner of the feasibility rule, so the CLI and the dashboard's
    sliders agree on what combinations are runnable.
    """
    start = max(min_train, n_draws - n_windows)
    return start, n_draws


def _score_window(predictions_by_position, actual_by_position, n_columns):
    """Set-based scoring: which numbers were guessed, not which slot they landed in."""
    main_pred = {predictions_by_position[p] for p in main_positions(n_columns)}
    main_actual = {actual_by_position[p] for p in main_positions(n_columns)}
    super_pos = super_position(n_columns)
    super_hit = predictions_by_position[super_pos] == actual_by_position[super_pos]
    return len(main_pred & main_actual), super_hit, len(main_pred)


def _run_windows(position_series, n_columns, n_windows, min_train, predict_window):
    """Drive the walk-forward loop and score whatever `predict_window` returns.

    `predict_window(position_series, t)` returns {model_name: {position: prediction}},
    or None for a model that could not predict that window. A window without a
    prediction is skipped outright — never scored against the actual draw, which
    would hand the model free hits.
    """
    total = len(next(iter(position_series.values())))
    start, total = window_bounds(total, n_windows, min_train)
    if start >= total:
        raise ValueError(
            f"No windows to evaluate: {total} draws with min_train={min_train}. "
            f"Lower min_train below {total}, or use a longer history."
        )

    rows = {name: [] for name in predict_window.model_names}
    for t in range(start, total):
        actual = {pos: int(frame.iloc[t]["y"]) for pos, frame in position_series.items()}
        for name, preds in predict_window(position_series, t).items():
            if preds is None:
                continue
            main_hits, super_hit, m_guessed = _score_window(preds, actual, n_columns)
            rows[name].append({"t": t, "main_hits": main_hits, "super_hit": super_hit,
                               "m_guessed": m_guessed})

    return {name: pd.DataFrame(rows[name], columns=RESULT_COLUMNS) for name in rows}


def _predictor(*model_names):
    """Tag a window predictor with the models it produces, so the name is written once."""
    def decorate(fn):
        fn.model_names = model_names
        return fn
    return decorate


def _statsforecast_window(n_columns):
    @_predictor(*MODEL_NAMES)
    def predict(position_series, t):
        truncated = {pos: frame.iloc[:t] for pos, frame in position_series.items()}
        forecast = fit_predict_all(truncated, h=1)

        preds_by_model = {}
        for name in MODEL_NAMES:
            clipped = adjusted_predictions(forecast, n_columns, model_name=name)
            preds_by_model[name] = dict(zip(clipped["unique_id"].astype(int), clipped["yhat_adjusted"]))
        return preds_by_model
    return predict


def _xgboost_window(n_columns):
    @_predictor("XGBoost")
    def predict(position_series, t):
        preds = {}
        for pos, frame in position_series.items():
            # Rows up to and including t: train_predict_one_step trains on all but
            # the last row, so y at t is never seen during training.
            yhat = train_predict_one_step(frame.iloc[: t + 1], pos, n_columns)
            if yhat is None:
                return {"XGBoost": None}
            preds[pos] = yhat
        return {"XGBoost": preds}
    return predict


def _prophet_window(n_columns):
    from Prophet import define_and_fit_model, predict_at_dates

    @_predictor("Prophet")
    def predict(position_series, t):
        preds = {}
        for pos, frame in position_series.items():
            model = define_and_fit_model(frame.iloc[:t])
            forecast = predict_at_dates(model, pos, n_columns, [frame.iloc[t]["ds"]])
            preds[pos] = int(forecast["yhat_adjusted"].iloc[0])
        return {"Prophet": preds}
    return predict


@_predictor("FrequencyBaseline")
def _frequency_window(position_series, t):
    return {"FrequencyBaseline": most_frequent_pick(position_series, upto=t)}


def summarize(results_by_model, alpha=0.05):
    """One row per model, with both the naive and the multiplicity-corrected verdict.

    A backtest scores k models against the same held-out draws, so it gets k
    chances at a false positive: at alpha=0.05 with six models there is a ~26%
    chance that at least one signal-free model clears the bar. `beats_chance`
    is the naive per-test verdict and is the one that misleads;
    `beats_chance_corrected` applies a Bonferroni threshold of alpha/k and is
    the column to read. This mirrors analysis/tickets.py:compare_strategies —
    the two tables answer the same question and must not disagree on how they
    handle it.
    """
    corrected = alpha / max(len(results_by_model), 1)
    summary = []
    for name, df in results_by_model.items():
        chance_test = beats_chance_test(df["main_hits"], df["m_guessed"])
        beats = chance_test["p_value_greater"]
        summary.append({
            "model": name,
            "n_windows": len(df),
            "avg_main_hits": df["main_hits"].mean(),
            "chance_avg_main_hits": chance_test["chance_mean"],
            "z_vs_chance": chance_test["z"],
            "p_value_better_than_chance": beats,
            "beats_chance": bool(beats < alpha) if pd.notna(beats) else False,
            "bonferroni_threshold": corrected,
            "beats_chance_corrected": bool(beats < corrected) if pd.notna(beats) else False,
            "super_hit_rate": df["super_hit"].mean(),
            "chance_super_hit_rate": expected_super_match_rate(),
        })
    return pd.DataFrame(summary).sort_values("avg_main_hits", ascending=False).reset_index(drop=True)


def run_all(position_series, n_columns, n_windows=15, min_train=60, include_prophet=False):
    predictors = [
        _frequency_window,
        _statsforecast_window(n_columns),
        _xgboost_window(n_columns),
    ]
    if include_prophet:
        predictors.append(_prophet_window(n_columns))

    results = {}
    for predict_window in predictors:
        t0 = time.time()
        results.update(_run_windows(position_series, n_columns, n_windows, min_train, predict_window))
        print(f"{', '.join(predict_window.model_names)} done in {time.time() - t0:.1f}s")
    return results



# ------------------------------------------------------------------ holdout by date
#
# The walk-forward loop above answers "how would this model do if I retrained
# it before every draw?". A date cutoff answers a different, more tangible
# question: train on everything up to July, then predict the draws that have
# already happened since. Both are offered because they are not the same
# experiment and the gap between them is itself informative.
#
#   expanding — refit at every held-out draw on all data before it. Optimistic
#               in the sense that the model keeps learning; it is what you would
#               actually do if you played every draw.
#   frozen    — fit once at the cutoff and forecast the whole remaining horizon
#               in one shot. Nothing after the cutoff is ever seen. This is the
#               literal "train to July, predict August and September" test, and
#               it is the harder of the two.
#
# Scoring is identical in both modes, so the summary tables are comparable.

HOLDOUT_MODES = ("expanding", "frozen")


def cutoff_bounds(dates, cutoff):
    """(n_train, n_holdout) for a date cutoff — draws on the cutoff day count as training."""
    ds = pd.to_datetime(pd.Series(list(dates))).sort_values()
    n_train = int((ds <= pd.Timestamp(cutoff)).sum())
    return n_train, len(ds) - n_train


def _frozen_statsforecast(position_series, n_columns, start, horizon):
    truncated = {pos: frame.iloc[:start] for pos, frame in position_series.items()}
    forecast = fit_predict_all(truncated, h=horizon)

    out = {}
    for name in MODEL_NAMES:
        clipped = adjusted_predictions(forecast, n_columns, model_name=name).sort_values("ds")
        by_step = {}
        for pos, group in clipped.groupby(clipped["unique_id"].astype(int)):
            for step, value in enumerate(group["yhat_adjusted"]):
                by_step.setdefault(step, {})[pos] = int(value)
        out[name] = by_step
    return out


def _frozen_xgboost(position_series, n_columns, start, horizon):
    by_step = {}
    for pos, frame in position_series.items():
        future_dates = frame.iloc[start:start + horizon]["ds"]
        predictions = forecast_horizon(frame.iloc[:start], pos, n_columns, future_dates)
        for step, value in enumerate(predictions):
            by_step.setdefault(step, {})[pos] = int(value)
    return {"XGBoost": by_step}


def _frozen_prophet(position_series, n_columns, start, horizon):
    from Prophet import define_and_fit_model, predict_at_dates

    by_step = {}
    for pos, frame in position_series.items():
        model = define_and_fit_model(frame.iloc[:start])
        forecast = predict_at_dates(model, pos, n_columns, frame.iloc[start:start + horizon]["ds"])
        for step, value in enumerate(forecast["yhat_adjusted"]):
            by_step.setdefault(step, {})[pos] = int(value)
    return {"Prophet": by_step}


def _frozen_frequency(position_series, n_columns, start, horizon):
    """The hottest number per slot as of the cutoff — the same pick for every held-out draw."""
    pick = most_frequent_pick(position_series, upto=start)
    return {"FrequencyBaseline": {step: pick for step in range(horizon)}}


def _score_frozen(predictions_by_step, position_series, n_columns, start):
    rows = []
    for step, preds in sorted(predictions_by_step.items()):
        t = start + step
        actual = {pos: int(frame.iloc[t]["y"]) for pos, frame in position_series.items()}
        main_hits, super_hit, m_guessed = _score_window(preds, actual, n_columns)
        rows.append({"t": t, "main_hits": main_hits, "super_hit": super_hit, "m_guessed": m_guessed})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def run_holdout(position_series, n_columns, cutoff, mode="expanding", include_prophet=False):
    """Train on draws up to `cutoff`, predict every draw after it, score against what happened.

    Returns (results_by_model, info). `info` carries the cutoff, the mode and
    the two split sizes so callers can report the experiment alongside its
    result instead of re-deriving it.
    """
    if mode not in HOLDOUT_MODES:
        raise ValueError(f"mode must be one of {HOLDOUT_MODES}, got {mode!r}")

    dates = next(iter(position_series.values()))["ds"]
    n_train, n_holdout = cutoff_bounds(dates, cutoff)
    if n_holdout < 1:
        raise ValueError(
            f"No draws after {pd.Timestamp(cutoff):%Y-%m-%d} — nothing to predict. "
            f"The history ends on {pd.Timestamp(dates.max()):%Y-%m-%d}; pick an earlier cutoff."
        )
    if n_train < MIN_TRAIN_FOR_HOLDOUT:
        raise ValueError(
            f"Only {n_train} draws on or before {pd.Timestamp(cutoff):%Y-%m-%d}; the models need at "
            f"least {MIN_TRAIN_FOR_HOLDOUT} to train on. Pick a later cutoff."
        )

    info = {"cutoff": pd.Timestamp(cutoff), "mode": mode, "n_train": n_train,
            "n_holdout": n_holdout,
            "holdout_start": pd.Timestamp(dates.iloc[n_train]),
            "holdout_end": pd.Timestamp(dates.iloc[-1])}

    if mode == "expanding":
        # Same loop as run_all, with the split pinned to the cutoff instead of a window count.
        return run_all(position_series, n_columns, n_windows=n_holdout, min_train=n_train,
                       include_prophet=include_prophet), info

    builders = [_frozen_frequency, _frozen_statsforecast, _frozen_xgboost]
    if include_prophet:
        builders.append(_frozen_prophet)

    results = {}
    for build in builders:
        t0 = time.time()
        by_model = build(position_series, n_columns, n_train, n_holdout)
        for name, by_step in by_model.items():
            results[name] = _score_frozen(by_step, position_series, n_columns, n_train)
        print(f"{', '.join(by_model)} done in {time.time() - t0:.1f}s")
    return results, info


def holdout_detail(results_by_model, position_series, n_columns):
    """Per-draw results: one row per held-out draw, one column per model.

    The summary says whether a model beat chance; this says what actually
    happened on 12 August. Both matter — the first is the verdict, the second
    is what makes it concrete.
    """
    dates = next(iter(position_series.values()))["ds"]
    rows = {}
    for name, df in results_by_model.items():
        for _, r in df.iterrows():
            t = int(r["t"])
            row = rows.setdefault(t, {"ds": dates.iloc[t]})
            row[f"{name} aciertos"] = int(r["main_hits"])
            row[f"{name} superbalota"] = bool(r["super_hit"])

    for t, row in rows.items():
        row["sorteo"] = " - ".join(
            str(int(position_series[p].iloc[t]["y"])) for p in main_positions(n_columns)
        )
        row["superbalota"] = int(position_series[super_position(n_columns)].iloc[t]["y"])

    detail = pd.DataFrame(sorted(rows.values(), key=lambda r: r["ds"]))
    lead = [c for c in ("ds", "sorteo", "superbalota") if c in detail.columns]
    return detail[lead + [c for c in detail.columns if c not in lead]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-windows", type=int, default=15)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--include-prophet", action="store_true")
    parser.add_argument("--cutoff", metavar="YYYY-MM-DD",
                        help="hold out every draw after this date instead of the last --n-windows")
    parser.add_argument("--mode", choices=HOLDOUT_MODES, default="expanding",
                        help="with --cutoff: refit before each held-out draw (expanding) or fit "
                             "once at the cutoff and forecast the whole horizon (frozen)")
    parser.add_argument("--current-format-only", action="store_true",
                        help="drop draws from before the 2017 rule change (6 balls from 1-45)")
    args = parser.parse_args()

    df, balls_expanded = load_and_preprocess(args.file, current_format_only=args.current_format_only)
    position_series = build_position_series(df, balls_expanded)
    n_columns = balls_expanded.shape[1]
    pd.set_option("display.width", 140)

    if args.cutoff:
        results, info = run_holdout(position_series, n_columns, args.cutoff, mode=args.mode,
                                    include_prophet=args.include_prophet)
        print(f"\n=== Holdout ({info['mode']}): trained on {info['n_train']} draws up to "
              f"{info['cutoff']:%Y-%m-%d}, predicting {info['n_holdout']} draws "
              f"{info['holdout_start']:%Y-%m-%d} to {info['holdout_end']:%Y-%m-%d} ===")
        detail = holdout_detail(results, position_series, n_columns)
        print(detail.to_string(index=False))
        detail.to_csv("holdout_detail.csv", index=False)
        out = "holdout_summary.csv"
    else:
        results = run_all(position_series, n_columns, args.n_windows, args.min_train,
                          args.include_prophet)
        out = "backtest_summary.csv"

    summary = summarize(results)
    print("\n=== Summary (does the model beat pure chance?) ===")
    print(summary.to_string(index=False))
    print("\nRead beats_chance_corrected, not beats_chance: several models are tested against the "
          "same draws, so the naive 5% bar is cleared by luck far more often than 5% of the time.")
    summary.to_csv(out, index=False)
