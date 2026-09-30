"""Baloto's evaluations as background jobs — the domain half of `core/jobs.py`.

`core/jobs.py` knows how to queue, run and store a job; this module says what
the lottery's jobs are: the walk-forward backtest and the date holdout, the two
surfaces that refit every model per window and used to block the dashboard
behind a spinner.

The draw history travels with the job as one frame, `ds` plus one column per
ball position. Position columns are renamed `b0`, `b1`, ... because a stored
column named `0` comes back as the string `'0'` — a different column in a
codebase whose position semantics are the thing it is most careful about.
Nothing here changes what a backtest computes: the handler rebuilds the same
`position_series` the page would have and calls the same `run_all` /
`run_holdout`, so a result read from the queue is the result the button used
to produce, manifest included.
"""

import pandas as pd

import lottery.backtest as bt
from lottery.models.common import build_position_series

WALK_FORWARD = "lottery.walk_forward"
HOLDOUT = "lottery.holdout"


def draws_input(df, balls_expanded):
    """The job input for a draw history: `ds` and one `b{position}` column per ball."""
    frame = pd.DataFrame({"ds": pd.to_datetime(df["ds"]).reset_index(drop=True)})
    for position in balls_expanded.columns:
        frame[f"b{position}"] = balls_expanded[position].reset_index(drop=True).astype("int64")
    return frame


def _position_series(draws):
    columns = sorted((c for c in draws.columns if c.startswith("b")), key=lambda c: int(c[1:]))
    balls_expanded = draws[columns].copy()
    balls_expanded.columns = [int(c[1:]) for c in columns]
    return build_position_series(draws[["ds"]], balls_expanded), len(columns)


def walk_forward(params, inputs, progress):
    """`run_all` over the last N draws, summarised as the page shows it."""
    position_series, n_columns = _position_series(inputs["draws"])
    progress(f"Refitting every model over {params['n_windows']} windows")
    results = bt.run_all(position_series, n_columns, n_windows=int(params["n_windows"]),
                         min_train=int(params["min_train"]),
                         include_prophet=bool(params.get("include_prophet", False)),
                         include_timesfm=bool(params.get("include_timesfm", False)))
    return {"summary": bt.summarize(results)}


def holdout(params, inputs, progress):
    """`run_holdout` at a date, with the per-draw detail the page draws."""
    position_series, n_columns = _position_series(inputs["draws"])
    progress(f"Training up to {params['cutoff']} ({params['mode']})")
    results, info = bt.run_holdout(position_series, n_columns, pd.Timestamp(params["cutoff"]),
                                   mode=params["mode"],
                                   include_prophet=bool(params.get("include_prophet", False)),
                                   include_timesfm=bool(params.get("include_timesfm", False)))
    return {"summary": bt.summarize(results),
            "detail": bt.holdout_detail(results, position_series, n_columns),
            "info": info}


HANDLERS = {WALK_FORWARD: walk_forward, HOLDOUT: holdout}
