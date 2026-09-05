"""How a history is split into training and held-out parts.

Both splits are pure index arithmetic over an ordered sequence of
observations — draws, matches, races — so neither knows anything about the
domain being evaluated.
"""

import pandas as pd


def window_bounds(n_observations, n_windows, min_train):
    """The (start, total) range a walk-forward backtest would evaluate.

    Single owner of the feasibility rule, so a CLI and a dashboard slider
    agree on what combinations are runnable. `start >= total` means the
    request is impossible and the caller should say so rather than silently
    evaluating nothing.
    """
    start = max(min_train, n_observations - n_windows)
    return start, n_observations


def cutoff_bounds(dates, cutoff):
    """(n_train, n_holdout) for a date cutoff.

    Observations dated on the cutoff day count as training: the cutoff reads
    as "train on everything up to and including this date". The dates need
    not arrive sorted.
    """
    ds = pd.to_datetime(pd.Series(list(dates))).sort_values()
    n_train = int((ds <= pd.Timestamp(cutoff)).sum())
    return n_train, len(ds) - n_train
