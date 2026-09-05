"""Planted-bias sensitivity — lottery/analysis/sensitivity.py.

The load-bearing invariant here is `independent_seeds`. Driving the draw
generator and the ticket generator from one seed makes the numbers drawn and
the numbers played come out of the same stream — a real ticket/draw
dependence, which is exactly what a lottery test hunts for. That mistake
reported a 17.5% false-positive rate on bias-free data and cost a full round
of investigation aimed at the wrong module. The regression test below plants
that bug deliberately and shows the detector reacting to it.
"""

import numpy as np
import pytest

from lottery.analysis.sensitivity import (
    DEFAULT_ALPHA,
    DEFAULT_FAVORED,
    DEFAULT_STRENGTHS,
    DETECTORS,
    biased_draws,
    detection_rate,
    independent_seeds,
    load_biased_and_preprocess,
    measured_favored_share,
    sensitivity_report,
    sensitivity_threshold,
)
from lottery.analysis.randomness import pooled_uniformity_test
from lottery.models.common import MAIN_POOL, main_positions
from lottery.utils.processor import format_violations


# ------------------------------------------------------- independent_seeds

def test_independent_seeds_returns_distinct_streams():
    seeds = independent_seeds(0, n=2)
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]


def test_the_derived_seeds_are_not_the_seed_itself():
    """The bug was passing the loop variable to both generators."""
    for seed in range(5):
        assert seed not in independent_seeds(seed)


def test_derived_streams_do_not_overlap():
    a, b = independent_seeds(0)
    first = np.random.default_rng(a).integers(0, 2**31, size=50)
    second = np.random.default_rng(b).integers(0, 2**31, size=50)
    assert not np.array_equal(first, second)


def test_independent_seeds_is_deterministic():
    assert independent_seeds(7) == independent_seeds(7)
    assert independent_seeds(7) != independent_seeds(8)


def test_sharing_one_stream_couples_draws_and_tickets():
    """The regression this module was shaped around, reproduced directly.

    With one `default_rng` feeding both the draw and the ticket, the ticket is
    drawn from the continuation of the very stream that produced the draw. Here
    the two are built from the *same* seed and compared against the independent
    pairing: the shared-stream version produces a systematically different match
    rate, which is the artifact that looked like an edge.
    """
    from lottery.analysis.tickets import draw_from_row, random_ticket

    def mean_matches(shared):
        hits = []
        for seed in range(60):
            data_seed, ticket_seed = (seed, seed) if shared else independent_seeds(seed)
            _, balls = load_biased_and_preprocess(n_draws=20, strength=0.0, seed=data_seed)
            rng = np.random.default_rng(ticket_seed)
            main_drawn, _ = draw_from_row(balls.iloc[-1])
            hits.append(len(random_ticket(rng).main_set & frozenset(main_drawn)))
        return float(np.mean(hits))

    assert mean_matches(shared=True) != mean_matches(shared=False)


# ------------------------------------------------------ the planted bias

def test_zero_strength_reproduces_uniform_draws():
    """`strength = 0` has to be a true control, not merely a weak bias."""
    _, balls = load_biased_and_preprocess(n_draws=2000, strength=0.0, seed=1)
    measured = measured_favored_share(balls, DEFAULT_FAVORED)
    assert measured["uniform_share"] == pytest.approx(len(DEFAULT_FAVORED) / MAIN_POOL)
    assert measured["favored_share"] == pytest.approx(measured["uniform_share"], abs=0.015)


def test_stronger_bias_means_a_larger_favoured_share():
    shares = []
    for strength in (0.0, 0.5, 1.0, 2.0):
        _, balls = load_biased_and_preprocess(n_draws=1500, strength=strength, seed=2)
        shares.append(measured_favored_share(balls, DEFAULT_FAVORED)["favored_share"])
    assert shares == sorted(shares)
    assert shares[-1] > shares[0] * 1.5


def test_the_bias_is_actually_visible_to_the_pooled_test():
    """If the planted signal were undetectable in principle, every row would be noise."""
    _, balls = load_biased_and_preprocess(n_draws=1500, strength=2.0, seed=3)
    assert pooled_uniformity_test(balls, main_positions(balls.shape[1]))["p_value"] < 0.001


def test_biased_draws_still_satisfy_the_data_contract():
    """The detectors must not be able to tell this from a scraped file except statistically."""
    df, balls = load_biased_and_preprocess(n_draws=200, strength=1.0, seed=4)
    assert not format_violations(balls).any()
    assert balls.shape[1] == 6
    assert len(df) == len(balls) == 200


def test_biased_draws_refuses_a_negative_strength():
    with pytest.raises(ValueError, match="non-negative"):
        biased_draws(strength=-0.5)


def test_biased_draws_are_reproducible():
    a = biased_draws(n_draws=20, strength=1.0, seed=9)
    b = biased_draws(n_draws=20, strength=1.0, seed=9)
    assert a.equals(b)


# ---------------------------------------------------------- the detectors

def test_the_control_strength_is_in_the_default_grid():
    assert DEFAULT_STRENGTHS[0] == 0.0, "0.0 is the control arm and is not optional"


def test_detection_rate_rejects_an_unknown_detector():
    with pytest.raises(ValueError, match="Unknown detector"):
        detection_rate(0.5, detector="vibes", n_draws=50, n_seeds=1)


@pytest.mark.parametrize("detector", sorted(DETECTORS))
def test_every_detector_reports_the_agreed_shape(detector):
    result = detection_rate(0.0, detector=detector, n_draws=120, n_seeds=2,
                            n_draws_back=20, min_history=50)
    assert result["detector"] == detector
    assert result["n_seeds"] == 2
    assert 0.0 <= result["detection_rate"] <= 1.0
    assert result["times_detected"] == result["detection_rate"] * 2


@pytest.mark.slow
def test_pooled_detects_a_strong_planted_bias():
    result = detection_rate(2.0, detector="pooled", n_draws=500, n_seeds=10)
    assert result["detection_rate"] >= 0.8


@pytest.mark.slow
def test_pooled_stays_near_alpha_on_the_control():
    result = detection_rate(0.0, detector="pooled", n_draws=500, n_seeds=20)
    assert result["detection_rate"] <= 0.25, "a control firing often means the harness is broken"


@pytest.mark.slow
def test_the_random_detector_is_blind_to_the_bias_by_construction():
    """A uniform ticket's expected matches do not depend on how the draw is weighted.

    `random` must stay near alpha even on heavily biased data. That is the
    control arm, not a failure — it is what makes the other two columns
    trustworthy.
    """
    result = detection_rate(2.0, detector="random", n_draws=400, n_seeds=20,
                            n_draws_back=100, min_history=200)
    assert result["detection_rate"] <= 0.25


@pytest.mark.slow
def test_sensitivity_report_covers_the_whole_grid():
    report = sensitivity_report(strengths=(0.0, 2.0), detectors=("pooled",),
                                n_draws=300, n_seeds=4)
    assert len(report) == 2
    assert {"detector", "strength", "detection_rate", "favored_share"} <= set(report.columns)
    by_strength = dict(zip(report["strength"], report["detection_rate"]))
    assert by_strength[2.0] > by_strength[0.0]


@pytest.mark.slow
def test_sensitivity_threshold_reports_where_detection_crosses_the_target():
    report = sensitivity_report(strengths=(0.0, 0.5, 1.0, 2.0), detectors=("pooled",),
                                n_draws=500, n_seeds=6)
    result = sensitivity_threshold(report, "pooled", target_rate=0.8)
    assert result is not None, "pooled should reach 80% somewhere in this range"
    assert result["detection_rate"] >= 0.8
    assert result["favored_share"] > result["uniform_share"]

    # The smallest qualifying strength, not merely any of them.
    qualifying = report[(report["detector"] == "pooled") & (report["detection_rate"] >= 0.8)]
    assert result["strength"] == qualifying["strength"].min()


@pytest.mark.slow
def test_an_undetectable_range_returns_none_rather_than_a_number():
    """None is a real answer: it means the null result rules nothing out."""
    report = sensitivity_report(strengths=(0.0,), detectors=("random",),
                                n_draws=300, n_seeds=4, n_draws_back=50, min_history=150)
    assert sensitivity_threshold(report, "random", target_rate=0.8) is None
