"""Pooling a model with the market — football/ensemble.py.

The invariant the whole module rests on is the pair of endpoints: weight 0 must
be the market *exactly* and weight 1 the model *exactly*. That is what makes the
blend's backtest readable — a blend at weight 0 scores identically to the
market, so any improvement as weight rises is the model adding information the
price did not already have. If the endpoints drift, that reading is gone and the
comparison stops meaning anything.
"""

import numpy as np
import pytest

from football.ensemble import POOLS, blend, linear_blend, logarithmic_pool

MODEL = np.array([[0.60, 0.25, 0.15], [0.20, 0.30, 0.50]])
MARKET = np.array([[0.45, 0.28, 0.27], [0.33, 0.30, 0.37]])


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_weight_zero_is_exactly_the_market(pool):
    assert np.allclose(blend(MODEL, MARKET, weight=0.0, pool=pool), MARKET)


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_weight_one_is_exactly_the_model(pool):
    assert np.allclose(blend(MODEL, MARKET, weight=1.0, pool=pool), MODEL)


@pytest.mark.parametrize("pool", sorted(POOLS))
@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_every_blend_is_a_distribution(pool, weight):
    out = blend(MODEL, MARKET, weight=weight, pool=pool)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.parametrize("pool", sorted(POOLS))
def test_agreement_is_a_fixed_point(pool):
    # Two sources saying the same thing cannot be pooled into something else.
    assert np.allclose(blend(MODEL, MODEL, weight=0.3, pool=pool), MODEL)


def test_linear_pooling_hedges():
    # A linear blend always lands between its inputs, component by component:
    # it can never be more confident than the more confident source.
    out = linear_blend(MODEL, MARKET, weight=0.5)
    lower, upper = np.minimum(MODEL, MARKET), np.maximum(MODEL, MARKET)
    assert ((out >= lower - 1e-12) & (out <= upper + 1e-12)).all()


def test_logarithmic_pooling_leans_harder_on_what_both_favour():
    """The first half of what distinguishes the two rules, and why both are offered.

    Both sources favour the home win. The log pool puts more on it than the
    linear pool does, and correspondingly less on the longshot — it still lands
    between the two inputs, it just sits nearer the confident end.
    """
    model = np.array([[0.70, 0.20, 0.10]])
    market = np.array([[0.65, 0.22, 0.13]])
    log_pooled = logarithmic_pool(model, market, weight=0.5)[0]
    linear_pooled = linear_blend(model, market, weight=0.5)[0]
    assert log_pooled[0] > linear_pooled[0]   # more on the agreed favourite
    assert log_pooled[2] < linear_pooled[2]   # less on the longshot


def test_logarithmic_pooling_is_harsh_on_what_one_source_nearly_ruled_out():
    """The second half, and the reason a conclusion can flip between the rules.

    One source almost vetoes an outcome the other thinks is even money. Linear
    pooling averages that away to something respectable; log pooling treats a
    near-veto as close to a veto. That is a real disagreement about how to
    combine evidence, not a rounding difference — which is why the pool used has
    to be reported alongside any result.
    """
    model = np.array([[0.50, 0.30, 0.20]])
    market = np.array([[0.02, 0.48, 0.50]])
    log_pooled = logarithmic_pool(model, market, weight=0.5)[0, 0]
    linear_pooled = linear_blend(model, market, weight=0.5)[0, 0]
    assert linear_pooled == pytest.approx(0.26)
    assert log_pooled < linear_pooled / 2


def test_a_zero_price_does_not_veto_the_model():
    # A market that priced an outcome at zero would otherwise take a logarithm
    # of zero and silently NaN out the whole row.
    market = np.array([[0.0, 0.40, 0.60]])
    out = logarithmic_pool(np.array([[0.5, 0.3, 0.2]]), market, weight=0.5)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)
    assert out[0, 0] > 0


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_a_weight_outside_zero_to_one_is_refused(weight):
    with pytest.raises(ValueError, match="between 0 and 1"):
        linear_blend(MODEL, MARKET, weight=weight)


def test_an_unknown_pool_is_refused():
    with pytest.raises(ValueError, match="Unknown pool"):
        blend(MODEL, MARKET, pool="harmonic")
