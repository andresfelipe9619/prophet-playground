"""Synthetic seasons — football/sample_data.py.

Unlike real data, this has a known right answer, and the tests exploit that:
the generative truth is computed exactly rather than simulated, and the
generated odds must round-trip back to it once the margin is removed. If that
round-trip breaks, either the de-margining or the generator is wrong, and the
first model built on top would inherit the fault invisibly.
"""

import numpy as np
import pytest

from football.common import ODDS_COLUMNS, OUTCOMES, PROBABILITY_COLUMNS
from football.market import implied_probabilities, market_probabilities, overround
from football.processor import check_match_format
from football.sample_data import (
    DEFAULT_MARGIN,
    HOME_ADVANTAGE,
    expected_goals,
    generate_matches,
    load_sample_and_preprocess,
    outcome_probabilities,
    round_robin,
    team_strengths,
)

TRUTH_COLUMNS = ["p_true_home", "p_true_draw", "p_true_away"]


# ----------------------------------------------------------- the generator

def test_team_strengths_are_centred():
    """A centred league keeps the overall goal rate at the intercept."""
    strengths = team_strengths(20, seed=0)
    assert strengths["attack"].mean() == pytest.approx(0.0, abs=1e-12)
    assert strengths["defence"].mean() == pytest.approx(0.0, abs=1e-12)
    assert len(strengths) == 20


def test_home_advantage_raises_the_home_rate():
    strengths = team_strengths(4, seed=0)
    home_rate, away_rate = expected_goals(strengths, "Team 01", "Team 01", HOME_ADVANTAGE)
    assert home_rate > away_rate
    assert home_rate / away_rate == pytest.approx(np.exp(HOME_ADVANTAGE))


def test_a_stronger_attack_scores_more():
    strengths = team_strengths(20, seed=0)
    best = strengths["attack"].idxmax()
    worst = strengths["attack"].idxmin()
    opponent = strengths.index[0]
    assert (expected_goals(strengths, best, opponent)[0]
            > expected_goals(strengths, worst, opponent)[0])


def test_round_robin_plays_everyone_home_and_away():
    fixtures = round_robin(["A", "B", "C"])
    assert len(fixtures) == 3 * 2
    assert ("A", "B") in fixtures and ("B", "A") in fixtures
    assert not any(h == a for h, a in fixtures)


def test_a_season_has_the_right_number_of_matches():
    assert len(load_sample_and_preprocess(n_teams=20, seed=0)) == 380
    assert len(load_sample_and_preprocess(n_teams=10, seed=0)) == 90


def test_generation_is_reproducible():
    assert generate_matches(n_teams=6, seed=3).equals(generate_matches(n_teams=6, seed=3))


def test_different_seeds_give_different_seasons():
    assert not generate_matches(n_teams=6, seed=1).equals(generate_matches(n_teams=6, seed=2))


# ------------------------------------------------------------- the truth

def test_the_true_probabilities_are_a_valid_distribution():
    matches = load_sample_and_preprocess(seed=0)
    truth = matches[TRUTH_COLUMNS].to_numpy()
    assert np.allclose(truth.sum(axis=1), 1.0)
    assert (truth > 0).all()


def test_outcome_probabilities_are_summed_not_simulated():
    """Exact, so a test comparing a forecast against them measures the forecast."""
    exact = outcome_probabilities(1.5, 1.2)
    assert exact.sum() == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    home, away = rng.poisson(1.5, 400_000), rng.poisson(1.2, 400_000)
    simulated = np.array([(home > away).mean(), (home == away).mean(), (home < away).mean()])
    assert exact == pytest.approx(simulated, abs=0.005)


def test_equal_teams_have_no_home_or_away_asymmetry_beyond_the_advantage():
    balanced = outcome_probabilities(1.4, 1.4)
    assert balanced[0] == pytest.approx(balanced[2], abs=1e-12), "no advantage, no asymmetry"


def test_a_higher_scoring_game_is_less_likely_to_be_drawn():
    assert outcome_probabilities(3.0, 3.0)[1] < outcome_probabilities(0.8, 0.8)[1]


# ------------------------------------------------------------ the market

def test_the_generated_odds_carry_the_intended_margin():
    matches = load_sample_and_preprocess(seed=0)
    margins = [overround(row) for row in matches[list(ODDS_COLUMNS)].to_numpy()]
    assert np.mean(margins) == pytest.approx(DEFAULT_MARGIN, abs=0.005)


def test_a_noiseless_market_round_trips_back_to_the_truth():
    """The check that ties the generator and the de-margining together.

    At `market_noise = 0` the simulated book prices the truth exactly, so
    removing the margin has to recover it. The residual is the 2-decimal
    rounding real odds also carry, not a modelling error.
    """
    matches = market_probabilities(load_sample_and_preprocess(seed=0, market_noise=0.0))
    recovered = matches[list(PROBABILITY_COLUMNS)].to_numpy()
    truth = matches[TRUTH_COLUMNS].to_numpy()
    assert np.abs(recovered - truth).max() < 0.005


def test_market_noise_moves_the_market_away_from_the_truth():
    def distance(noise):
        matches = market_probabilities(load_sample_and_preprocess(seed=0, market_noise=noise))
        return np.abs(matches[list(PROBABILITY_COLUMNS)].to_numpy()
                      - matches[TRUTH_COLUMNS].to_numpy()).mean()

    assert distance(0.0) < distance(0.15) < distance(0.40)


def test_a_noisy_market_is_still_a_valid_distribution():
    matches = market_probabilities(load_sample_and_preprocess(seed=0, market_noise=0.5))
    assert np.allclose(matches[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0)


def test_odds_can_be_written_into_the_opening_columns():
    """So the opening-versus-closing guard can be tested on realistic-looking data."""
    closing = load_sample_and_preprocess(seed=0, closing_odds=True)
    opening = load_sample_and_preprocess(seed=0, closing_odds=False)
    assert closing.attrs["odds_are_closing"] is True
    assert opening.attrs["odds_are_closing"] is False
    assert opening.attrs["odds_source"] == "market_opening_average"


def test_a_season_can_be_generated_without_odds():
    matches = load_sample_and_preprocess(seed=0, include_odds=False)
    assert matches.attrs["odds_source"] is None
    assert matches[list(ODDS_COLUMNS)].isna().all().all()


# ------------------------------------------------------------- realism

def test_the_data_goes_through_the_real_contract():
    """No in-memory shortcut: every test using this data exercises the parser."""
    matches = load_sample_and_preprocess(seed=0)
    assert check_match_format(matches) is None
    assert matches["ds"].is_monotonic_increasing
    assert set(matches["outcome"]) <= set(OUTCOMES)


def test_the_outcome_mix_looks_like_a_real_league():
    """Calibrated against the Premier League: ~45% home, ~25% draw, ~30% away.

    Data that does not look like football would make every downstream test
    easier to pass and less informative.
    """
    matches = load_sample_and_preprocess(n_teams=20, seed=0)
    share = matches["outcome"].value_counts(normalize=True)
    assert share["H"] == pytest.approx(0.45, abs=0.06)
    assert share["D"] == pytest.approx(0.25, abs=0.06)
    assert share["A"] == pytest.approx(0.30, abs=0.06)


def test_the_scoring_rate_looks_like_a_real_league():
    matches = load_sample_and_preprocess(n_teams=20, seed=0)
    assert matches["home_goals"].mean() == pytest.approx(1.5, abs=0.3)
    assert matches["away_goals"].mean() == pytest.approx(1.2, abs=0.3)
    assert matches["home_goals"].mean() > matches["away_goals"].mean()


def test_home_teams_win_more_than_away_teams():
    matches = load_sample_and_preprocess(seed=0)
    share = matches["outcome"].value_counts(normalize=True)
    assert share["H"] > share["A"]


def test_the_market_is_calibrated_against_what_actually_happened():
    """The realistic setting a model faces: the book is right on average.

    Averaged over a season, the market's home-win probability has to land near
    the observed home-win rate. A market that is systematically off would make
    it trivially beatable, and every later result would be an artefact of the
    generator rather than a finding.
    """
    matches = market_probabilities(load_sample_and_preprocess(n_teams=20, seed=0))
    for column, outcome in zip(PROBABILITY_COLUMNS, OUTCOMES):
        predicted = matches[column].mean()
        observed = (matches["outcome"] == outcome).mean()
        assert predicted == pytest.approx(observed, abs=0.05), outcome


def test_the_answer_key_is_not_part_of_the_contract():
    """`p_true_*` do not exist on real data; nothing outside tests may read them."""
    from football.processor import MATCH_COLUMNS
    assert not any(c.startswith("p_true") for c in MATCH_COLUMNS)
