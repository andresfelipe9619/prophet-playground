"""Walk-forward evaluation of a football model against the closing line.

The football counterpart of lottery/backtest.py, and it shares the shape on
purpose. `run_all` holds out the last N matches and refits Dixon-Coles before
each one (expanding window, one step ahead). `run_holdout` holds out everything
after a date, either refitting per match (`expanding`) or fitting once at the
cutoff (`frozen`) — the frozen mode is the fast, concrete "train in March,
predict the rest of the season" run.

Both paths end in the same place: `football.evaluation.beats_market_test` over
every (model probs, market probs, outcome) triple collected, so the summaries
are directly comparable. A window whose held-out fixture involves a team the
training slice never saw is skipped, never scored — exactly as the lottery
skips a window a model could not predict.

Fitting Dixon-Coles per window is the slow part; `--n-windows` defaults low and
`--cutoff ... --mode frozen` avoids the refit loop entirely.
"""

import argparse
import os

import numpy as np
import pandas as pd

from core.windows import cutoff_bounds, window_bounds
from football.common import ODDS_COLUMNS, PROBABILITY_COLUMNS
from football.dixon_coles import DixonColes, UnknownTeamError
from football.evaluation import beats_market_test
from football.market import market_probabilities

MIN_TRAIN = 100


def _market_row_probs(match_row, method):
    """Market probability vector for one match row, NaN if the row has no price."""
    frame = pd.DataFrame([match_row])
    if frame[list(ODDS_COLUMNS)].isna().to_numpy().any():
        return np.array([np.nan, np.nan, np.nan])
    out = market_probabilities(frame, method=method)
    return out[list(PROBABILITY_COLUMNS)].to_numpy()[0]


def _score(model_probs, market_probs, outcomes, metric, n_comparisons=1):
    return beats_market_test(np.array(model_probs), np.array(market_probs), outcomes,
                             metric=metric, n_comparisons=n_comparisons)


def _collect(train_for, matches, indices, half_life, method, frozen_model=None):
    """Fit-and-predict over `indices`, returning the three aligned lists plus a skip count."""
    model_probs, market_probs, outcomes = [], [], []
    skipped = 0
    for t in indices:
        test = matches.iloc[t]
        try:
            model = frozen_model if frozen_model is not None else \
                DixonColes.fit(train_for(t), half_life=half_life)
            p_model = model.predict_outcome(test["home_team"], test["away_team"])
        except UnknownTeamError:
            skipped += 1
            continue
        model_probs.append(p_model)
        market_probs.append(_market_row_probs(test, method))
        outcomes.append(test["outcome"])
    return model_probs, market_probs, outcomes, skipped


def run_all(matches, n_windows=30, min_train=MIN_TRAIN, half_life=None,
            method="multiplicative", metric="rps"):
    """Hold out the last `n_windows` matches, refitting Dixon-Coles before each."""
    matches = matches.sort_values("ds").reset_index(drop=True)
    start, total = window_bounds(len(matches), n_windows, min_train)
    model_probs, market_probs, outcomes, skipped = _collect(
        lambda t: matches.iloc[:t], matches, range(start, total), half_life, method)

    result = _score(model_probs, market_probs, outcomes, metric)
    result.update({"n_windows_scored": len(outcomes), "n_windows_skipped": skipped,
                   "mode": "expanding_last_n", "method": method, "half_life": half_life})
    return result


def run_holdout(matches, cutoff, mode="expanding", half_life=None,
                method="multiplicative", metric="rps"):
    """Hold out every match after `cutoff`.

    `mode='expanding'` refits before each held-out match; `mode='frozen'` fits
    once at the cutoff and forecasts the whole remaining horizon.
    """
    if mode not in ("expanding", "frozen"):
        raise ValueError(f"mode must be 'expanding' or 'frozen', got {mode!r}")

    matches = matches.sort_values("ds").reset_index(drop=True)
    n_train, n_holdout = cutoff_bounds(matches["ds"], cutoff)

    frozen_model = None
    if mode == "frozen":
        frozen_model = DixonColes.fit(matches.iloc[:n_train], half_life=half_life)

    model_probs, market_probs, outcomes, skipped = _collect(
        lambda t: matches.iloc[:t], matches, range(n_train, n_train + n_holdout),
        half_life, method, frozen_model=frozen_model)

    result = _score(model_probs, market_probs, outcomes, metric)
    result.update({"n_windows_scored": len(outcomes), "n_windows_skipped": skipped,
                   "mode": mode, "method": method, "half_life": half_life})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", required=True,
                        help="comma-separated CSV filenames inside --data-dir")
    parser.add_argument("--data-dir", default="exported_data/football")
    parser.add_argument("--n-windows", type=int, default=30)
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    parser.add_argument("--cutoff", default=None, help="ISO date; hold out everything after it")
    parser.add_argument("--mode", choices=("expanding", "frozen"), default="expanding")
    parser.add_argument("--half-life", type=float, default=None, help="days; time-decay weighting")
    parser.add_argument("--method", choices=("multiplicative", "additive", "power"),
                        default="multiplicative")
    parser.add_argument("--metric", choices=("brier", "rps", "log_loss"), default="rps")
    parser.add_argument("--extra", action="store_true", help="load via the extra-file contract")
    parser.add_argument("--league", default=None, help="league to pick from an --extra file")
    args = parser.parse_args()

    paths = [os.path.join(args.data_dir, name) for name in args.seasons.split(",")]
    if args.extra:
        from football.extra_processor import load_extra
        frames = [load_extra(p, league=args.league) for p in paths]
        matches = pd.concat(frames, ignore_index=True).sort_values("ds").reset_index(drop=True)
    else:
        from football.processor import load_seasons
        matches = load_seasons(paths)

    if args.cutoff:
        result = run_holdout(matches, cutoff=args.cutoff, mode=args.mode,
                             half_life=args.half_life, method=args.method, metric=args.metric)
    else:
        result = run_all(matches, n_windows=args.n_windows, min_train=args.min_train,
                         half_life=args.half_life, method=args.method, metric=args.metric)

    for key, value in result.items():
        print(f"{key:>24}: {value}")


if __name__ == "__main__":
    main()
