"""Canaries: does the evaluation machinery ever see the answer it is scoring?

Every domain here has conventions against leakage — chronological splits, an
`as_of` cutoff, "results strictly before the race" — and none of them is
*proved* by anything. A convention is a thing a reviewer can check; a canary
is a thing the suite can.

The shape is `lottery/analysis/sensitivity.py`'s, turned around. That module
plants a signal and measures that the detectors fire; these plant the *absence*
of one and measure that they stay quiet. Two kinds:

**Shuffled targets.** Permute the outcomes in time and refit. The relationship
between what a model can see and what it is scored on is gone, so no verdict
may survive — and one that does is the pipeline reading the held-out row. This
is the strongest available statement, because it needs no knowledge of how the
split is implemented.

**Cutoffs.** A function taking `as_of` must return exactly what it would have
returned if everything from `as_of` onward had never been recorded. Two tests
per function: delete the future and compare, then add an absurd future and
compare. The second is the one that catches a `<=` where a `<` was meant.

**And positive controls for both**, because a canary that cannot fire is
decoration. Each kind gets a deliberately leaking counterpart that the same
machinery must catch.

Why these and not more: a leak that survives both a shuffle and a deleted
future is not a leak in the split, it is a bug in the score, which is what
`tests/test_football_evaluation.py` and `tests/test_cycling_evaluation.py`
already pin with their zero-effect endpoints.
"""

import numpy as np
import pandas as pd
import pytest

from cycling.baseline import form_worths
from cycling.evaluation import compare_forecasters
from cycling.plackett_luce import PlackettLuce
from cycling.processor import preprocess_results
from cycling.sample_data import generate_stage_race
from football.backtest import compare_models
from football.common import ODDS_COLUMNS, outcome_from_goals
from football.h2h import head_to_head, team_form
from football.processor import preprocess_matches
from football.sample_data import generate_matches

# ---------------------------------------------------------------- fixtures


def _matches(seed=0, n_teams=14, market_noise=0.6):
    raw = generate_matches(n_teams=n_teams, seed=seed, market_noise=market_noise).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


def _shuffle_results_in_time(matches, seed):
    """Move every result **and its prices together** to a different fixture.

    The calendar and the team labels stay; the result-plus-odds block moves as
    one. Two teams' names now say nothing about how their match went, so a
    model fitted on the past has nothing to find — while the market, which
    travelled with its own result, is still pricing it correctly.

    Moving the odds with the result is the whole design, and the first version
    of this canary got it wrong. Shuffling the outcomes alone leaves the market
    pricing results that no longer happened, which makes it *actively* wrong;
    Elo then converges on the base rates, beats a confidently mispriced book,
    and the canary fires on a pipeline with nothing wrong with it. Keeping the
    pair intact leaves a sharp market and a blind model, which is the only
    arrangement in which "the model won" can mean one thing.
    """
    rng = np.random.default_rng(seed)
    out = matches.copy()
    order = rng.permutation(len(out))
    for column in ("home_goals", "away_goals", *ODDS_COLUMNS):
        out[column] = matches[column].to_numpy()[order]
    out["outcome"] = [outcome_from_goals(h, a)
                      for h, a in zip(out["home_goals"], out["away_goals"], strict=True)]
    out.attrs = dict(matches.attrs)
    return out


def _race(seed=0, n_stages=10):
    raw = generate_stage_race(n_stages=n_stages, seed=seed,
                              climbing_stages=set(range(1, n_stages + 1)))
    return preprocess_results(raw, validate=False)


def _shuffle_riders_within_each_race(results, seed):
    """Reassign each race's finishing positions to its riders at random.

    Ranks, times and statuses keep their distribution race by race; which rider
    got which is now noise. A model that still beats the ranking has seen the
    race it is predicting.
    """
    rng = np.random.default_rng(seed)
    out = results.copy()
    for _, index in out.groupby(["race", "kind", "stage"], dropna=False).groups.items():
        riders = out.loc[index, "rider"].to_numpy()
        out.loc[index, "rider"] = rng.permutation(riders)
    return out


def _straddles_zero(row):
    """The interval covers no effect — the shape of an honest null result."""
    return bool(row["ci_low"] <= 0 <= row["ci_high"])


# ------------------------------------------------- 1 · shuffled-target canaries


@pytest.mark.slow
def test_football_finds_nothing_once_the_results_are_shuffled_in_time():
    """The canary. A leaking walk-forward beats the market even on noise."""
    shuffled = _shuffle_results_in_time(_matches(seed=4, n_teams=16), seed=11)
    table = compare_models(shuffled, n_windows=120, min_train=160,
                           models=("dixon_coles", "elo"))

    assert len(table) == 2
    for _, row in table.iterrows():
        assert not row["beats_market_corrected"], (
            f"{row['model']} beat the market on shuffled results — "
            "the walk-forward is seeing the match it is scoring")
        assert _straddles_zero(row), (
            f"{row['model']}'s interval excludes zero on shuffled results")


@pytest.mark.slow
def test_cycling_finds_nothing_once_the_riders_are_shuffled_within_each_race():
    shuffled = _shuffle_riders_within_each_race(_race(seed=2), seed=7)

    def ranking(history, riders, as_of):
        return form_worths(history, riders, as_of=as_of)

    def model(history, riders, as_of):
        if not len(history):
            return None
        return PlackettLuce.fit(history).worths_for(riders)

    table, _ = compare_forecasters(shuffled, {"ranking": ranking, "model": model},
                                   baseline="ranking", min_history=3)
    row = table[table["forecaster"] == "model"].iloc[0]
    assert not row["beats_baseline_corrected"], (
        "the model beat the ranking on shuffled riders — the walk-forward is "
        "seeing the race it is forecasting")
    assert _straddles_zero(row)


# ------------------------------------------------------- positive controls
#
# A canary that cannot fire is not evidence, it is decoration. These plant the
# leak deliberately and require the same machinery to catch it — the other half
# of `lottery/analysis/sensitivity.py`'s argument, where synthetic i.i.d. data
# proves the detectors do not cry wolf and only a planted signal proves they can
# hear one.
#
# The cycling control is the stronger of the two: `compare_forecasters` takes
# forecasters as arguments, so the leak can be planted inside the real
# walk-forward and the whole pipeline is on trial. `compare_models` takes model
# *names*, so football's control is planted one level down, at the test that
# delivers the verdict. Between them: cycling shows the walk can catch a leak,
# football shows the verdict can.


@pytest.mark.slow
def test_the_cycling_canary_fires_on_a_forecaster_that_reads_the_race():
    shuffled = _shuffle_riders_within_each_race(_race(seed=2), seed=7)

    def ranking(history, riders, as_of):
        return form_worths(history, riders, as_of=as_of)

    def leaks(history, riders, as_of):
        """Reads the race it is forecasting — the bug the canary exists for."""
        race = shuffled[shuffled["ds"] == as_of].sort_values("rank")
        placed = {rider: place for place, rider in enumerate(race["rider"])}
        return np.array([1.0 / (1 + placed.get(rider, len(riders))) for rider in riders])

    table, _ = compare_forecasters(shuffled, {"ranking": ranking, "leaks": leaks},
                                   baseline="ranking", min_history=3)
    row = table[table["forecaster"] == "leaks"].iloc[0]
    assert row["beats_baseline_corrected"], (
        "a forecaster reading the race it forecasts did not beat the ranking — "
        "the canary above proves nothing, because it cannot fail")


def test_the_football_verdict_fires_on_a_forecast_that_read_the_result():
    from football.common import OUTCOMES, PROBABILITY_COLUMNS
    from football.evaluation import beats_market_test
    from football.market import market_probabilities

    matches = _shuffle_results_in_time(_matches(seed=4, n_teams=16), seed=11).iloc[160:]
    market = market_probabilities(matches)[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    outcomes = list(matches["outcome"])

    column = {outcome: i for i, outcome in enumerate(OUTCOMES)}
    leaked = np.full((len(outcomes), 3), 0.05)
    for row, outcome in enumerate(outcomes):
        leaked[row, column[outcome]] = 0.90

    result = beats_market_test(leaked, market, outcomes)
    assert result["beats_market_corrected"] is True, (
        "a forecast that had seen the result did not beat the market — "
        "the shuffled-target canary cannot fail either")


# --------------------------------------------------------- 3 · cutoff canaries
#
# Cheap, exact, and the ones that actually fail when somebody writes `<=`.


@pytest.fixture
def cutoff_matches():
    matches = _matches(seed=1, n_teams=10)
    return matches, matches["ds"].iloc[len(matches) // 2]


def test_team_form_ignores_everything_from_the_cutoff_onward(cutoff_matches):
    matches, cutoff = cutoff_matches
    team = matches["home_team"].iloc[0]
    past_only = matches[matches["ds"] < cutoff]

    assert team_form(matches, team, as_of=cutoff) == team_form(past_only, team, as_of=cutoff)


def test_head_to_head_ignores_everything_from_the_cutoff_onward(cutoff_matches):
    matches, cutoff = cutoff_matches
    home, away = matches["home_team"].iloc[0], matches["away_team"].iloc[0]
    past_only = matches[matches["ds"] < cutoff]

    assert (head_to_head(matches, home, away, as_of=cutoff)
            == head_to_head(past_only, home, away, as_of=cutoff))


def test_a_match_played_on_the_cutoff_day_is_not_visible(cutoff_matches):
    """`as_of` reads as "before this", not "up to and including it". A fixture
    kicking off on the day you forecast is not information you had."""
    matches, cutoff = cutoff_matches
    on_the_day = matches[matches["ds"] == cutoff]
    assert len(on_the_day), "fixture chose a date with no match on it"

    team = on_the_day["home_team"].iloc[0]
    without = matches[matches["ds"] != cutoff]
    assert team_form(matches, team, as_of=cutoff) == team_form(without, team, as_of=cutoff)


def test_an_absurd_future_match_changes_nothing(cutoff_matches):
    """The other direction: adding the future must not move a past-only answer.

    Deleting the future and comparing can pass while a function quietly reads
    a column it should not; inventing one that no correct answer could contain
    cannot."""
    matches, cutoff = cutoff_matches
    team = matches["home_team"].iloc[0]
    opponent = matches.loc[matches["home_team"] == team, "away_team"].iloc[0]

    invented = matches.iloc[[0]].copy()
    invented["ds"] = matches["ds"].max() + pd.Timedelta(days=365)
    invented["home_team"], invented["away_team"] = team, opponent
    invented["home_goals"], invented["away_goals"] = 99, 0
    invented["outcome"] = "H"
    with_future = pd.concat([matches, invented], ignore_index=True)

    assert team_form(with_future, team, as_of=cutoff) == team_form(matches, team, as_of=cutoff)
    assert (head_to_head(with_future, team, opponent, as_of=cutoff)
            == head_to_head(matches, team, opponent, as_of=cutoff))


@pytest.fixture
def cutoff_race():
    results = _race(seed=3, n_stages=8)
    stages = sorted(results["ds"].unique())
    return results, stages[len(stages) // 2]


def test_form_worths_ignores_everything_from_the_cutoff_onward(cutoff_race):
    results, cutoff = cutoff_race
    riders = sorted(results["rider"].unique())[:12]
    past_only = results[results["ds"] < cutoff]

    np.testing.assert_allclose(form_worths(results, riders, as_of=cutoff),
                               form_worths(past_only, riders, as_of=cutoff))


def test_form_worths_cannot_see_the_stage_it_is_forecasting(cutoff_race):
    """The stage *on* the cutoff is the one being predicted. This is the test
    that fails if `<` ever becomes `<=`."""
    results, cutoff = cutoff_race
    riders = sorted(results["rider"].unique())[:12]
    without_that_stage = results[results["ds"] != cutoff]

    np.testing.assert_allclose(form_worths(results, riders, as_of=cutoff),
                               form_worths(without_that_stage, riders, as_of=cutoff))


def test_an_absurd_future_stage_changes_nothing(cutoff_race):
    results, cutoff = cutoff_race
    riders = sorted(results["rider"].unique())[:12]

    invented = results[results["ds"] == results["ds"].max()].copy()
    invented["ds"] = results["ds"].max() + pd.Timedelta(days=400)
    invented["stage"] = 99
    # Reverse the finishing order, so a leak shows up as a different answer
    # rather than the same one arrived at twice.
    invented["rank"] = invented["rank"].to_numpy()[::-1]
    with_future = pd.concat([results, invented], ignore_index=True)

    np.testing.assert_allclose(form_worths(with_future, riders, as_of=cutoff),
                               form_worths(results, riders, as_of=cutoff))
