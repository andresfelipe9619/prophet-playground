"""Walk-forward backtest: does any model actually beat pure chance?

For each of the last `n_windows` real draws, every model is trained only on
data available before that draw (expanding window, 1-step ahead) and scored
on how many of the 5 main balls it matched and whether it matched the
superbalota. Those hit counts are then compared against the exact
hypergeometric chance baseline (models/baseline.py) with a z-test — this is
the number that should decide whether a model is worth trusting, since raw
"N matches" means nothing without knowing what pure luck would already give
you.

Prophet is included but off by default (`--include-prophet`) because it
refits per position per window and is far slower than the other models.
"""

import argparse
import time

import pandas as pd

from models.baseline import beats_chance_test, expected_super_match_rate, most_frequent_pick
from models.common import DEFAULT_DATA_PATH, build_position_series, main_positions, super_position
from models.statsforecast_model import MODEL_NAMES, adjusted_predictions, fit_predict_all
from models.xgboost_model import train_predict_one_step
from utils.processor import load_and_preprocess

RESULT_COLUMNS = ["t", "main_hits", "super_hit", "m_guessed"]


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


def summarize(results_by_model):
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
            "beats_chance_p<0.05": bool(beats < 0.05) if pd.notna(beats) else False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-windows", type=int, default=15)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--include-prophet", action="store_true")
    args = parser.parse_args()

    df, balls_expanded = load_and_preprocess(args.file)
    position_series = build_position_series(df, balls_expanded)

    results = run_all(position_series, balls_expanded.shape[1], args.n_windows, args.min_train,
                      args.include_prophet)
    summary = summarize(results)
    pd.set_option("display.width", 120)
    print("\n=== Backtest summary (does the model beat pure chance?) ===")
    print(summary.to_string(index=False))
    summary.to_csv("backtest_summary.csv", index=False)
