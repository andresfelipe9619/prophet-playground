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

from models.baseline import beats_chance_test, expected_main_matches, expected_super_match_rate
from models.common import build_position_series, clip_to_range
from models.statsforecast_model import MODEL_NAMES, fit_predict_all
from models.xgboost_model import train_predict_one_step
from utils.processor import load_and_preprocess

MAIN_POSITIONS_END = 5  # positions 0..4 are main balls, position 5 is the superbalota


def _score_window(predictions_by_position, actual_by_position, n_super_position):
    main_pred = {predictions_by_position[p] for p in range(n_super_position)}
    main_actual = {actual_by_position[p] for p in range(n_super_position)}
    main_hits = len(main_pred & main_actual)
    super_hit = predictions_by_position[n_super_position] == actual_by_position[n_super_position]
    return main_hits, super_hit, len(main_pred)


def backtest_statsforecast(position_series, n_columns, n_windows, min_train):
    total = len(next(iter(position_series.values())))
    start = max(min_train, total - n_windows)
    rows = {name: [] for name in MODEL_NAMES}

    for t in range(start, total):
        truncated = {pos: frame.iloc[:t] for pos, frame in position_series.items()}
        actual = {pos: int(frame.iloc[t]["y"]) for pos, frame in position_series.items()}
        forecast = fit_predict_all(truncated, h=1)

        for name in MODEL_NAMES:
            preds = {}
            for pos in range(n_columns):
                raw = forecast.loc[forecast["unique_id"] == pos, name].iloc[0]
                preds[pos] = clip_to_range(raw, pos, n_columns)
            main_hits, super_hit, m_guessed = _score_window(preds, actual, MAIN_POSITIONS_END)
            rows[name].append({"t": t, "main_hits": main_hits, "super_hit": super_hit, "m_guessed": m_guessed})

    return {name: pd.DataFrame(rows[name]) for name in MODEL_NAMES}


def backtest_xgboost(position_series, n_columns, n_windows, min_train):
    total = len(next(iter(position_series.values())))
    start = max(min_train, total - n_windows)
    rows = []

    for t in range(start, total):
        preds, actual = {}, {}
        for pos, frame in position_series.items():
            truncated = frame.iloc[: t + 1]
            actual[pos] = int(frame.iloc[t]["y"])
            yhat = train_predict_one_step(truncated, pos, n_columns)
            preds[pos] = yhat if yhat is not None else actual[pos]  # fallback if not enough history yet
        main_hits, super_hit, m_guessed = _score_window(preds, actual, MAIN_POSITIONS_END)
        rows.append({"t": t, "main_hits": main_hits, "super_hit": super_hit, "m_guessed": m_guessed})

    return pd.DataFrame(rows)


def backtest_prophet(position_series, n_columns, n_windows, min_train):
    from Prophet import define_and_fit_model, predict_at_dates

    total = len(next(iter(position_series.values())))
    start = max(min_train, total - n_windows)
    rows = []

    for t in range(start, total):
        preds, actual = {}, {}
        for pos, frame in position_series.items():
            train = frame.iloc[:t]
            actual[pos] = int(frame.iloc[t]["y"])
            model = define_and_fit_model(train)
            forecast = predict_at_dates(model, pos, n_columns, [frame.iloc[t]["ds"]])
            preds[pos] = int(forecast["yhat_adjusted"].iloc[0])
        main_hits, super_hit, m_guessed = _score_window(preds, actual, MAIN_POSITIONS_END)
        rows.append({"t": t, "main_hits": main_hits, "super_hit": super_hit, "m_guessed": m_guessed})

    return pd.DataFrame(rows)


def backtest_frequency_baseline(position_series, n_columns, n_windows, min_train):
    total = len(next(iter(position_series.values())))
    start = max(min_train, total - n_windows)
    rows = []

    for t in range(start, total):
        preds, actual = {}, {}
        for pos, frame in position_series.items():
            train = frame.iloc[:t]
            actual[pos] = int(frame.iloc[t]["y"])
            preds[pos] = int(train["y"].mode().iloc[0])  # historically most frequent number so far
        main_hits, super_hit, m_guessed = _score_window(preds, actual, MAIN_POSITIONS_END)
        rows.append({"t": t, "main_hits": main_hits, "super_hit": super_hit, "m_guessed": m_guessed})

    return pd.DataFrame(rows)


def summarize(results_by_model):
    summary = []
    for name, df in results_by_model.items():
        chance_test = beats_chance_test(df["main_hits"], df["m_guessed"])
        summary.append({
            "model": name,
            "n_windows": len(df),
            "avg_main_hits": df["main_hits"].mean(),
            "chance_avg_main_hits": chance_test["chance_mean"],
            "z_vs_chance": chance_test["z"],
            "p_value_vs_chance": chance_test["p_value"],
            "beats_chance_p<0.05": (chance_test["p_value"] < 0.05) if pd.notna(chance_test["p_value"]) else False,
            "super_hit_rate": df["super_hit"].mean(),
            "chance_super_hit_rate": expected_super_match_rate(),
        })
    return pd.DataFrame(summary).sort_values("avg_main_hits", ascending=False).reset_index(drop=True)


def run_all(position_series, n_columns, n_windows=15, min_train=60, include_prophet=False):
    results = {}

    t0 = time.time()
    results["FrequencyBaseline"] = backtest_frequency_baseline(position_series, n_columns, n_windows, min_train)
    print(f"FrequencyBaseline done in {time.time() - t0:.1f}s")

    t0 = time.time()
    results.update(backtest_statsforecast(position_series, n_columns, n_windows, min_train))
    print(f"StatsForecast models done in {time.time() - t0:.1f}s")

    t0 = time.time()
    results["XGBoost"] = backtest_xgboost(position_series, n_columns, n_windows, min_train)
    print(f"XGBoost done in {time.time() - t0:.1f}s")

    if include_prophet:
        t0 = time.time()
        results["Prophet"] = backtest_prophet(position_series, n_columns, n_windows, min_train)
        print(f"Prophet done in {time.time() - t0:.1f}s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="exported_data/final-final.csv")
    parser.add_argument("--n-windows", type=int, default=15)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--include-prophet", action="store_true")
    args = parser.parse_args()

    df, balls_expanded = load_and_preprocess(args.file)
    position_series = build_position_series(df, balls_expanded)
    n_columns = balls_expanded.shape[1]

    results = run_all(position_series, n_columns, args.n_windows, args.min_train, args.include_prophet)
    summary = summarize(results)
    pd.set_option("display.width", 120)
    print("\n=== Backtest summary (does the model beat pure chance?) ===")
    print(summary.to_string(index=False))
    summary.to_csv("backtest_summary.csv", index=False)
