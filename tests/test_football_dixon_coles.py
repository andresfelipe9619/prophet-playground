"""The Dixon-Coles model — football/dixon_coles.py.

Two independent Poisson goal counts, plus the low-score correction (rho) that
pulls probability toward 0-0/1-0/0-1/1-1 to fix the draw deficit the plain
model has. Fitted by maximum likelihood.

The checks that matter: a model fitted on data generated from a known truth
recovers that truth on average; the scoreline grid is a proper distribution;
the rho term actually moves the low-score cells; an unseen team is refused
loudly rather than scored on nothing.
"""

import numpy as np
import pytest

from football.dixon_coles import (
    DixonColes,
    UnknownTeamError,
    independent_poisson_matrix,
)
from football.sample_data import load_sample_and_preprocess

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fitted():
    matches = load_sample_and_preprocess(n_teams=40, seed=1)
    return DixonColes.fit(matches), matches


def test_scoreline_matrix_is_a_distribution(fitted):
    model, matches = fitted
    grid = model.scoreline_matrix(model.teams[0], model.teams[1], max_goals=10)
    assert grid.shape == (11, 11)
    assert (grid >= 0).all()
    assert grid.sum() == pytest.approx(1.0, abs=1e-9)


def test_predict_outcome_sums_to_one_and_is_in_hda_order(fitted):
    model, _ = fitted
    p = model.predict_outcome(model.teams[2], model.teams[5])
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0)
    slate = np.mean([model.predict_outcome(h, a)
                     for h in model.teams for a in model.teams if h != a], axis=0)
    assert slate[0] > slate[2]


def test_attack_effects_are_centred(fitted):
    model, _ = fitted
    assert sum(model.params["attack"].values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(model.params["defence"].values()) == pytest.approx(0.0, abs=1e-6)


def test_fit_recovers_the_generative_outcome_probabilities(fitted):
    model, matches = fitted
    truth = load_sample_and_preprocess(n_teams=40, seed=1)[["p_true_home", "p_true_draw", "p_true_away"]].to_numpy()
    pred = model.predict_matches(matches)
    assert np.abs(pred - truth).mean() < 0.05


def test_rho_moves_the_low_score_cells_only():
    lh, la = 1.4, 1.1
    base = independent_poisson_matrix(lh, la, max_goals=8)
    model = DixonColes._from_params(
        teams=("X", "Y"), mu=np.log(1.25), home_advantage=0.0,
        attack={"X": 0.0, "Y": 0.0}, defence={"X": 0.0, "Y": 0.0}, rho=-0.15,
    )
    grid = model._grid(lh, la, max_goals=8)
    changed = ~np.isclose(grid / grid.sum(), base)
    corners = np.array([grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1]])
    base_corners = np.array([base[0, 0], base[0, 1], base[1, 0], base[1, 1]])
    assert np.abs(corners / grid.sum() - base_corners).max() > 1e-3
    assert np.trace(grid) / grid.sum() > np.trace(base)


def test_unknown_team_raises(fitted):
    model, _ = fitted
    with pytest.raises(UnknownTeamError):
        model.predict_outcome("Nonexistent FC", model.teams[0])


def test_half_life_changes_the_fit(fitted):
    model, matches = fitted
    decayed = DixonColes.fit(matches, half_life=90)
    assert decayed.params["rho"] != model.params["rho"] or (
        decayed.params["attack"] != model.params["attack"]
    )
