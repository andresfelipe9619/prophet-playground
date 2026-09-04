"""Synthetic Baloto-shaped draw data, so the dashboard and scripts are runnable
without the user's private exported_data/ CSVs.

Numbers are generated as genuinely independent uniform draws, which is
exactly the null hypothesis the analysis/randomness.py tests check for — so
running the dashboard against this demo data is also a good sanity check
that those tests correctly say "looks random" on data that is, by
construction, random.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from models.common import DRAW_WEEKDAYS, MAIN_BALL_RANGE, SUPER_BALL_RANGE


def generate_sample_draws(n_draws=400, start_date="2018-01-03", seed=42):
    rng = np.random.default_rng(seed)
    cursor = pd.to_datetime(start_date)
    rows = []
    while len(rows) < n_draws:
        if cursor.weekday() in DRAW_WEEKDAYS:
            # Deliberately NOT sorted: these are 5 independent uniform draws, one per
            # column position. Real official results are often published sorted
            # ascending, which turns each column into an order statistic instead of
            # a uniform draw (see analysis.randomness.is_sorted_ascending) — the demo
            # data stays unsorted so the randomness tests show the clean i.i.d. case.
            main = rng.choice(
                np.arange(MAIN_BALL_RANGE[0], MAIN_BALL_RANGE[1] + 1), size=5, replace=False
            )
            super_ball = rng.integers(SUPER_BALL_RANGE[0], SUPER_BALL_RANGE[1] + 1)
            ball_str = "-".join(str(n) for n in [*main, super_ball])
            rows.append({"Date": cursor.strftime("%d/%m/%Y"), "Ball": ball_str})
        cursor += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def load_sample_and_preprocess(n_draws=400, start_date="2018-01-03", seed=42):
    """Same return shape as utils.processor.load_and_preprocess, for drop-in use."""
    df = generate_sample_draws(n_draws=n_draws, start_date=start_date, seed=seed)
    df["ds"] = pd.to_datetime(df["Date"], dayfirst=True)
    balls_expanded = df["Ball"].str.split("-", expand=True).apply(pd.to_numeric)
    return df, balls_expanded
