"""Statistical analysis of how (un)predictable each ball position actually is.

This is the module that keeps the rest of the project honest: before trusting
any forecast, check whether there is any exploitable pattern at all. For a
fair lottery there shouldn't be, and these functions let the dashboard show
that explicitly instead of implying a model "found" something it didn't.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from models.common import max_for_position, min_for_position, series_label


def frequency_table(position_series, position, n_columns):
    low = min_for_position(position, n_columns)
    high = max_for_position(position, n_columns)
    numbers = np.arange(low, high + 1)
    counts = position_series["y"].value_counts().reindex(numbers, fill_value=0)
    n_draws = len(position_series)
    expected = n_draws / len(numbers)
    table = pd.DataFrame({
        "number": numbers,
        "count": counts.values,
        "expected_count": expected,
        "deviation_pct": (counts.values - expected) / expected * 100 if expected else 0,
    })
    return table.sort_values("number").reset_index(drop=True)


def chi_square_uniformity(freq_table):
    """Goodness-of-fit test: is this position drawn uniformly, or is something off?

    High p-value (say > 0.05) = no evidence against uniform randomness, which is
    the expected and "healthy" result for a fair lottery.
    """
    observed = freq_table["count"].to_numpy()
    stat, p_value = stats.chisquare(observed)
    return {"chi2": float(stat), "p_value": float(p_value), "n_categories": len(observed)}


def runs_test(sequence):
    """Wald-Wolfowitz runs test around the median: are ups/downs sequenced randomly?"""
    values = np.asarray(sequence, dtype=float)
    median = np.median(values)
    signs = values >= median
    signs = signs[values != median]  # drop ties at the median, standard practice
    if len(signs) < 2:
        return {"z": np.nan, "p_value": np.nan, "n_runs": 0}

    n1 = int(signs.sum())
    n2 = int((~signs).sum())
    n_runs = 1 + int(np.sum(signs[1:] != signs[:-1]))

    mean_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    if var_runs <= 0:
        return {"z": np.nan, "p_value": np.nan, "n_runs": n_runs}

    z = (n_runs - mean_runs) / np.sqrt(var_runs)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(p_value), "n_runs": n_runs}


def autocorrelation_check(series, n_lags=10):
    """ACF values plus a Ljung-Box test for autocorrelation up to n_lags.

    A low Ljung-Box p-value would be the one thing that could justify a
    time-series model here; for a fair lottery expect it to stay high.
    """
    values = pd.Series(series).astype(float)
    n_lags = min(n_lags, max(1, len(values) // 4))
    acf_values = acf(values, nlags=n_lags, fft=True)
    lb = acorr_ljungbox(values, lags=[n_lags], return_df=True)
    return {
        "acf": acf_values.tolist(),
        "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_p_value": float(lb["lb_pvalue"].iloc[0]),
        "n_lags": n_lags,
    }


def gap_table(position_series, position, n_columns):
    """Per-number gap stats: last seen, average/typical gap, and an 'overdue' z-score.

    The overdue score is the classic "cold number is due" heuristic. It is
    included because people ask for it, not because it has predictive value:
    for i.i.d. draws the time since a number last appeared carries no
    information about when it will appear next (gambler's fallacy).
    """
    low = min_for_position(position, n_columns)
    high = max_for_position(position, n_columns)
    last_ds = position_series["ds"].max()
    rows = []
    for number in range(low, high + 1):
        dates = position_series.loc[position_series["y"] == number, "ds"].sort_values()
        if len(dates) == 0:
            rows.append({
                "number": number, "times_seen": 0, "last_date": pd.NaT,
                "avg_gap_days": np.nan, "std_gap_days": np.nan,
                "days_since_last": np.nan, "overdue_score": np.nan,
            })
            continue
        gaps = dates.diff().dropna().dt.days
        last_date = dates.max()
        days_since_last = (last_ds - last_date).days
        avg_gap = gaps.mean() if len(gaps) else np.nan
        std_gap = gaps.std() if len(gaps) > 1 else np.nan
        overdue = (days_since_last - avg_gap) / std_gap if std_gap and std_gap > 0 else np.nan
        rows.append({
            "number": number, "times_seen": len(dates), "last_date": last_date,
            "avg_gap_days": avg_gap, "std_gap_days": std_gap,
            "days_since_last": days_since_last, "overdue_score": overdue,
        })
    return pd.DataFrame(rows).sort_values("number").reset_index(drop=True)


def hot_cold_numbers(position_series, position, n_columns, recent_draws=20):
    """Compare each number's share of the most recent draws vs. its all-time share."""
    freq_all = frequency_table(position_series, position, n_columns).set_index("number")["count"]
    recent = position_series.sort_values("ds").tail(recent_draws)
    freq_recent = recent["y"].value_counts().reindex(freq_all.index, fill_value=0)

    total_all = freq_all.sum()
    total_recent = freq_recent.sum()
    share_all = freq_all / total_all if total_all else freq_all * 0
    share_recent = freq_recent / total_recent if total_recent else freq_recent * 0

    table = pd.DataFrame({
        "number": freq_all.index,
        "recent_count": freq_recent.values,
        "share_recent_pct": (share_recent * 100).values,
        "share_all_time_pct": (share_all * 100).values,
        "delta_pct": ((share_recent - share_all) * 100).values,
    }).reset_index(drop=True)
    return table.sort_values("delta_pct", ascending=False).reset_index(drop=True)


def is_sorted_ascending(balls_expanded, main_positions=5, tolerance=0.95):
    """Heuristic: are the main balls stored sorted ascending within each draw?

    Official results are often published this way. If so, each column is an
    order statistic (min, 2nd-smallest, ...) rather than a uniform draw, and
    per-position chi-square tests below will show spurious "non-random"
    structure that reflects the sorting, not anything predictable. Use
    pooled_uniformity_test in that case instead of trusting per-position tests.
    """
    main = balls_expanded.iloc[:, :main_positions]
    is_sorted_row = (main.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
    return float(is_sorted_row.mean()) >= tolerance


def pooled_uniformity_test(balls_expanded, low, high, positions=None):
    """Chi-square uniformity test pooling several columns together (e.g. all 5 main balls).

    This is the position-agnostic, sort-order-proof version of
    chi_square_uniformity: it only asks "does every number in [low, high]
    show up equally often across all draws and slots combined?", which is
    the real question when individual positions may be order statistics.
    """
    columns = balls_expanded.iloc[:, positions] if positions is not None else balls_expanded
    pooled = columns.to_numpy().ravel()
    numbers = np.arange(low, high + 1)
    counts = pd.Series(pooled).value_counts().reindex(numbers, fill_value=0)
    stat, p_value = stats.chisquare(counts.to_numpy())
    return {"chi2": float(stat), "p_value": float(p_value), "n_categories": len(numbers), "n_observations": len(pooled)}


def randomness_report(position_series, position, n_columns):
    """One-shot summary the dashboard shows as the 'is this actually predictable?' verdict."""
    freq = frequency_table(position_series, position, n_columns)
    chi2 = chi_square_uniformity(freq)
    runs = runs_test(position_series["y"].to_numpy())
    autocorr = autocorrelation_check(position_series["y"].to_numpy())

    looks_random = (
        chi2["p_value"] > 0.05
        and (np.isnan(runs["p_value"]) or runs["p_value"] > 0.05)
        and autocorr["ljung_box_p_value"] > 0.05
    )

    return {
        "label": series_label(position, n_columns),
        "n_draws": len(position_series),
        "chi_square": chi2,
        "runs_test": runs,
        "autocorrelation": autocorr,
        "looks_random": looks_random,
    }
