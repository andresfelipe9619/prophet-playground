"""core/significance.py — tested without any lottery in sight.

These are the tests a second domain inherits. If they hold, the evaluation
discipline transfers; nothing here may need a ball, a draw or a pool to run.
"""

import numpy as np
import pytest

from core.significance import bonferroni_threshold, verdicts, z_test_against_null


def test_a_sample_from_the_null_does_not_beat_it():
    rng = np.random.default_rng(0)
    observed = rng.normal(loc=2.0, scale=1.0, size=500)
    result = z_test_against_null(observed, 2.0, 1.0)
    assert result["p_value_greater"] > 0.05
    assert result["observed_mean"] == pytest.approx(2.0, abs=0.15)
    assert result["null_mean"] == 2.0


def test_scoring_far_above_the_null_is_detected():
    result = z_test_against_null([3.0] * 200, 1.0, 1.0)
    assert result["z"] > 0
    assert result["p_value_greater"] < 1e-10


def test_scoring_far_below_the_null_must_not_read_as_beating_it():
    """The whole reason both p-values exist.

    A predictor significantly *worse* than the null gets a tiny two-sided
    p-value. Only `p_value_greater` may back a "beats the baseline" claim.
    """
    result = z_test_against_null([0.0] * 200, 1.0, 1.0)
    assert result["z"] < 0
    assert result["p_value"] < 1e-10
    assert result["p_value_greater"] > 0.99


def test_the_two_p_values_are_consistent():
    for observed in ([1.5] * 50, [0.5] * 50, [1.0] * 50):
        r = z_test_against_null(observed, 1.0, 1.0)
        expected_two_sided = 2 * min(r["p_value_greater"], 1 - r["p_value_greater"])
        assert r["p_value"] == pytest.approx(expected_two_sided, abs=1e-9)


def test_a_per_observation_null_is_accepted():
    """The null may shift between observations — that is why it is not a scalar."""
    observed = [1, 0, 1, 0]
    shifting = z_test_against_null(observed, [0.5, 1.0, 1.5, 2.0], [1.0, 1.0, 1.0, 1.0])
    flat = z_test_against_null(observed, 1.25, 1.0)
    assert shifting["null_mean"] == pytest.approx(1.25)
    assert shifting["z"] == pytest.approx(flat["z"])


def test_a_per_observation_variance_changes_the_standard_error():
    observed = [2.0] * 4
    tight = z_test_against_null(observed, 1.0, 0.25)
    loose = z_test_against_null(observed, 1.0, 4.0)
    assert tight["z"] > loose["z"], "less null variance makes the same gap more significant"


def test_no_observations_yields_nan_not_a_verdict():
    result = z_test_against_null([], 1.0, 1.0)
    assert np.isnan(result["z"])
    assert np.isnan(result["p_value_greater"])


def test_a_degenerate_null_refuses_to_invent_a_z():
    """Zero variance means no sampling distribution: report the means, not a z."""
    result = z_test_against_null([1.0, 1.0], 1.0, 0.0)
    assert np.isnan(result["z"])
    assert result["observed_mean"] == 1.0 and result["null_mean"] == 1.0


@pytest.mark.parametrize("k, expected", [(1, 0.05), (2, 0.025), (6, 0.05 / 6)])
def test_bonferroni_threshold(k, expected):
    assert bonferroni_threshold(0.05, k) == pytest.approx(expected)


def test_bonferroni_threshold_survives_an_empty_comparison_set():
    assert bonferroni_threshold(0.05, 0) == 0.05


def test_verdicts_returns_both_columns_together():
    """They come back as one dict so a surface cannot report one without the other."""
    result = verdicts(0.001, 0.05, 0.01)
    assert set(result) == {"beats_chance", "bonferroni_threshold", "beats_chance_corrected"}
    assert result["beats_chance"] is True
    assert result["beats_chance_corrected"] is True


def test_the_corrected_verdict_is_stricter():
    result = verdicts(0.03, 0.05, 0.01)
    assert result["beats_chance"] is True, "clears the naive 5% bar"
    assert result["beats_chance_corrected"] is False, "but not the corrected one"


def test_a_missing_p_value_is_not_a_pass():
    for p in (float("nan"), None):
        result = verdicts(p, 0.05, 0.01)
        assert result["beats_chance"] is False
        assert result["beats_chance_corrected"] is False


def test_core_imports_nothing_from_a_domain():
    """The seam only holds if it is one-directional.

    Checked on the import graph rather than the text, since the docstrings in
    `core/` legitimately name the lottery as the worked example.
    """
    import ast
    import pathlib

    domains = {"lottery", "dashboard", "scripts"}
    for path in sorted(pathlib.Path("core").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in domains, f"{path} imports {name}"
