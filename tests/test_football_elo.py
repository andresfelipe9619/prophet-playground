"""Elo ratings and the ordered-logit outcome map — football/elo.py.

Two things carry this module. The rating engine has to be classic Elo, which
means conserving total rating and moving in the right direction; and the map
from a rating gap to three probabilities has to be a genuine ordered
distribution, which means the cut points never cross and the draw sits between
the two wins rather than being sprinkled on afterwards.

The third thing these pin is the one a refactor breaks silently: ratings are
walked in **date** order, not row order, and every match is predicted from the
ratings as they stood before it.
"""

import numpy as np
import pandas as pd
import pytest

from football.common import OUTCOMES, UnknownTeamError, outcome_index
from football.elo import DEFAULT_RATING, Elo, expected_score
from football.processor import preprocess_matches
from football.sample_data import generate_matches


@pytest.fixture(scope="module")
def matches():
    raw = generate_matches(n_teams=16, seed=0).drop(columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


@pytest.fixture(scope="module")
def model(matches):
    return Elo.fit(matches)


def test_probabilities_are_a_distribution(model, matches):
    probs = model.predict_matches(matches)
    assert probs.shape == (len(matches), len(OUTCOMES))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_the_cut_points_never_cross(model):
    # An ordered logit is only a distribution while cut_low < cut_high, which is
    # why the gap is optimised in logs rather than checked for afterwards.
    assert model.cut_low < model.cut_high
    assert model.scale > 0


def test_total_rating_is_conserved(model):
    # Classic Elo is zero-sum: what the winner gains the loser loses. A drift
    # here means the update is being applied asymmetrically somewhere.
    total = sum(model.ratings.values())
    assert total == pytest.approx(DEFAULT_RATING * len(model.ratings))


def test_a_bigger_gap_means_a_bigger_home_probability(model):
    gaps = np.array([-400.0, -100.0, 0.0, 100.0, 400.0])
    from football.elo import _probabilities

    probs = _probabilities(gaps, model.cut_low, model.cut_high, model.scale)
    home = probs[:, outcome_index("H")]
    away = probs[:, outcome_index("A")]
    assert list(home) == sorted(home)          # rises with the gap
    assert list(away) == sorted(away, reverse=True)  # and falls


def test_the_draw_peaks_where_the_sides_are_even(model):
    from football.elo import _probabilities

    gaps = np.linspace(-600, 600, 61)
    draw = _probabilities(gaps, model.cut_low, model.cut_high, model.scale)[:, outcome_index("D")]
    # The most likely draw is near the middle of the range, not at an extreme.
    assert abs(gaps[int(np.argmax(draw))]) < 250


def test_the_strongest_team_tops_the_ranking(matches):
    """A side that wins every match must end up first, whatever else is going on."""
    extra = pd.DataFrame([
        {"ds": matches["ds"].max() + pd.Timedelta(days=d + 1),
         "home_team": "Invicto", "away_team": team,
         "home_goals": 5, "away_goals": 0, "outcome": "H"}
        for d, team in enumerate(sorted(set(matches["home_team"]))[:12])
    ])
    model = Elo.fit(pd.concat([matches, extra], ignore_index=True))
    assert model.ranking()[0][0] == "Invicto"


def test_expected_score_is_the_classic_logistic():
    assert expected_score(1500, 1500, home_advantage=0) == pytest.approx(0.5)
    assert expected_score(1900, 1500, home_advantage=0) == pytest.approx(10 / 11, abs=1e-3)
    # Home advantage shifts an even tie in the home side's favour, by definition.
    assert expected_score(1500, 1500) > 0.5


def test_row_order_does_not_change_the_model(matches):
    """Ratings are walked in date order, so a shuffled frame must fit identically.

    This is the test that notices a refactor dropping the sort: on real data,
    rows usually arrive in date order anyway, so nothing else would complain.
    """
    shuffled = matches.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a, b = Elo.fit(matches), Elo.fit(shuffled)
    assert a.ratings.keys() == b.ratings.keys()
    for team in a.ratings:
        assert a.ratings[team] == pytest.approx(b.ratings[team])
    assert a.cut_low == pytest.approx(b.cut_low)
    assert a.cut_high == pytest.approx(b.cut_high)


def test_an_unseen_team_is_refused_not_guessed(model):
    with pytest.raises(UnknownTeamError):
        model.predict_outcome("Real Inexistente", model.teams[0])


def test_a_tiny_frame_falls_back_instead_of_wandering():
    # Below the fit floor the optimiser has nothing to learn from; the starting
    # values stand, and they still have to be a valid distribution.
    tiny = preprocess_matches(
        generate_matches(n_teams=4, seed=3).drop(columns=["TrueH", "TrueD", "TrueA"]),
        validate=False).head(6)
    model = Elo.fit(tiny)
    probs = model.predict_outcome(tiny.iloc[0]["home_team"], tiny.iloc[0]["away_team"])
    assert probs.sum() == pytest.approx(1.0)
    assert model.cut_low < model.cut_high
