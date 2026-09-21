"""What running the bets feels like — football/bankroll.py

An edge is a number; a bankroll is a path, and the path is what decides whether
a person can actually run a system. These pin the three things that make the
picture honest: the null path exists and falls, the drawdown grows with the
stake, and a bettor with only the market's numbers never bets at all.
"""

import numpy as np
import pandas as pd
import pytest

from football.bankroll import (
    RUIN_THRESHOLD,
    drawdown_distribution,
    risk_of_ruin,
    roi_interval,
    simulate_bankroll,
    stake_fraction_sweep,
    summarise,
)
from football.common import ODDS_COLUMNS, PROBABILITY_COLUMNS
from football.market import market_probabilities
from football.processor import preprocess_matches
from football.sample_data import generate_matches
from football.value import kelly_fraction

TRUTH = ("TrueH", "TrueD", "TrueA")


@pytest.fixture(scope="module")
def priced():
    """Truth, de-margined market, raw odds and outcomes, all on the same rows."""
    raw = generate_matches(n_teams=16, seed=4, market_noise=0.6)
    truth = raw[list(TRUTH)].to_numpy(dtype=float)
    truth = truth / truth.sum(axis=1, keepdims=True)

    matches = market_probabilities(
        preprocess_matches(raw.drop(columns=list(TRUTH)), validate=False))
    usable = (matches[list(PROBABILITY_COLUMNS)].notna().all(axis=1)
              & matches[list(ODDS_COLUMNS)].notna().all(axis=1))
    keep = np.flatnonzero(usable.to_numpy())
    matches = matches[usable]

    return (truth[keep],
            matches[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float),
            matches[list(ODDS_COLUMNS)].to_numpy(dtype=float),
            list(matches["outcome"]))


# ------------------------------------------------------------- the two bars


def test_a_bettor_with_only_the_market_never_bets():
    """Not a quirk — value.py's two bars. A de-margined probability is compared
    against the raw price it must beat, and it never clears the margin. This is
    why the null path is built by redrawing the world rather than by re-staking
    from the market's own numbers."""
    priced = pd.DataFrame({"odds_home": [2.10], "odds_draw": [3.40], "odds_away": [3.80]})
    market = market_probabilities(priced)[list(PROBABILITY_COLUMNS)].to_numpy()
    stakes = kelly_fraction(market[0], priced[list(ODDS_COLUMNS)].to_numpy()[0], fraction=1.0)
    assert np.nan_to_num(stakes).max() == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------- the null path


def test_every_simulation_carries_a_null_path(priced):
    truth, market, odds, outcomes = priced
    simulation = simulate_bankroll(truth, market, odds, outcomes, n_paths=120, seed=1)
    assert simulation["paths"].shape == simulation["null_paths"].shape
    assert simulation["n_staked"] > 0


def test_the_null_bleeds_rather_than_holding_flat(priced):
    """The same bets in a world where the model knows nothing still pay the
    overround. A null that sits flat has forgotten the vig, which is the single
    most common way these charts lie."""
    truth, market, odds, outcomes = priced
    simulation = simulate_bankroll(truth, market, odds, outcomes, n_paths=300, seed=1)

    null = roi_interval(simulation["null_paths"])
    assert null["median_roi"] < 0
    assert null["share_losing"] > 0.5


def test_a_model_that_knows_the_truth_beats_its_own_null(priced):
    """The positive control. If real information does not separate from the
    null, the comparison the chart rests on is not measuring anything."""
    truth, market, odds, outcomes = priced
    simulation = simulate_bankroll(truth, market, odds, outcomes, n_paths=300, seed=1)

    summary = summarise(simulation)
    model = summary[summary["series"] == "model"].iloc[0]
    null = summary[summary["series"] == "null (market)"].iloc[0]
    assert model["median_roi"] > null["median_roi"]
    assert model["risk_of_ruin"] < null["risk_of_ruin"]


# ------------------------------------------------------- stake and survival


def test_drawdown_grows_with_the_stake(priced):
    """The picture of why value.py defaults to a quarter."""
    truth, market, odds, outcomes = priced
    sweep = stake_fraction_sweep(truth, market, odds, outcomes,
                                 fractions=(0.1, 0.25, 0.5, 1.0), n_paths=200, seed=1)
    assert list(sweep["median_drawdown"]) == sorted(sweep["median_drawdown"])
    assert list(sweep["risk_of_ruin"]) == sorted(sweep["risk_of_ruin"])


def test_full_kelly_ruins_even_a_model_that_is_exactly_right(priced):
    """Being right about a probability is not the same as surviving its
    variance. This is the measurement behind the quarter-Kelly default."""
    truth, market, odds, outcomes = priced
    sweep = stake_fraction_sweep(truth, market, odds, outcomes,
                                 fractions=(0.25, 1.0), n_paths=200, seed=1)
    quarter = sweep[sweep["stake_fraction"] == 0.25].iloc[0]
    full = sweep[sweep["stake_fraction"] == 1.0].iloc[0]

    assert full["risk_of_ruin"] > 0.9
    assert quarter["risk_of_ruin"] < full["risk_of_ruin"]
    assert full["median_roi"] < quarter["median_roi"]


# ---------------------------------------------------------------- the pieces


def test_a_drawdown_is_measured_from_the_running_peak():
    """What was given back from the best it ever looked — the moment a person
    actually stops — not the fall from where they started."""
    rose_then_fell = np.array([[1.0, 2.0, 1.0]])
    assert drawdown_distribution(rose_then_fell)["worst"] == pytest.approx(0.5)


def test_a_path_that_only_rises_has_no_drawdown():
    assert drawdown_distribution(np.array([[1.0, 1.5, 2.0]]))["worst"] == pytest.approx(0.0)


def test_ruin_counts_ever_not_only_at_the_end():
    """A system that dips to 8% and recovers has already lost its operator."""
    dipped = np.array([[1.0, RUIN_THRESHOLD / 2, 1.2]])
    assert risk_of_ruin(dipped) == pytest.approx(1.0)
    assert roi_interval(dipped)["median_roi"] > 0


def test_the_roi_interval_brackets_its_own_median():
    paths = np.cumprod(1 + np.random.default_rng(0).normal(0.01, 0.1, (200, 50)), axis=1)
    roi = roi_interval(paths)
    assert roi["ci_low"] <= roi["median_roi"] <= roi["ci_high"]


def test_without_bootstrap_every_path_is_the_same_history(priced):
    truth, market, odds, outcomes = priced
    simulation = simulate_bankroll(truth, market, odds, outcomes, n_paths=5, seed=1,
                                   bootstrap=False)
    assert np.allclose(simulation["paths"], simulation["paths"][0])
    # The null still varies: it redraws the world, which is the whole point.
    assert not np.allclose(simulation["null_paths"], simulation["null_paths"][0])
