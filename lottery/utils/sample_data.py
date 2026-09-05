"""Synthetic Baloto-shaped draw data, so the dashboard and scripts are runnable
without the user's private exported_data/ CSVs.

Numbers are generated as genuinely independent uniform draws, which is
exactly the null hypothesis the lottery/analysis/randomness.py tests check for — so
running the dashboard against this demo data is also a good sanity check
that those tests correctly say "looks random" on data that is, by
construction, random.
"""

import numpy as np
import pandas as pd

from lottery.models.common import MAIN_BALLS_DRAWN, MAIN_BALL_RANGE, SUPER_BALL_RANGE, next_draw_dates
from lottery.utils.processor import preprocess_draws


def generate_sample_draws(n_draws=400, start_date="2018-01-03", seed=42):
    rng = np.random.default_rng(seed)
    day_before = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    dates = next_draw_dates(day_before, n_draws)

    rows = []
    for date in dates:
        # Deliberately NOT sorted: these are independent uniform draws, one per
        # column position. Real official results are often published sorted
        # ascending, which turns each column into an order statistic instead of
        # a uniform draw (see lottery.analysis.randomness.is_sorted_ascending) — the demo
        # data stays unsorted so the randomness tests show the clean i.i.d. case.
        main = rng.choice(
            np.arange(MAIN_BALL_RANGE[0], MAIN_BALL_RANGE[1] + 1), size=MAIN_BALLS_DRAWN, replace=False
        )
        super_ball = rng.integers(SUPER_BALL_RANGE[0], SUPER_BALL_RANGE[1] + 1)
        rows.append({
            "Date": date.strftime("%d/%m/%Y"),
            "Ball": "-".join(str(n) for n in [*main, super_ball]),
        })
    return pd.DataFrame(rows)


def load_sample_and_preprocess(n_draws=400, start_date="2018-01-03", seed=42):
    """Same return shape as lottery.utils.processor.load_and_preprocess, for drop-in use."""
    return preprocess_draws(generate_sample_draws(n_draws=n_draws, start_date=start_date, seed=seed))
