"""The data contract and the two-eras detection — lottery/utils/processor.py.

Baloto changed rules in April 2017 (before: 6 balls from 1-45, no
superbalota). Both eras publish as six dash-separated numbers, so only the
values give the mix away. The tests below pin the deliberate difference
between `format_violations` (provably impossible today) and
`current_format_mask` (cuts at the date of the last violation).
"""

import warnings

import pandas as pd
import pytest

from tests.conftest import draws_frame, parsed
from lottery.utils.processor import (
    check_draw_format,
    current_format_mask,
    format_violations,
    load_and_preprocess,
    preprocess_draws,
)

CLEAN = [
    ("05/01/2019", "3-12-19-27-41-8"),
    ("07/01/2019", "1-5-9-14-22-16"),
    ("09/01/2019", "40-2-33-11-7-1"),
]


def test_requires_the_contract_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        preprocess_draws(pd.DataFrame({"Fecha": ["05/01/2019"], "Ball": ["1-2-3-4-5-6"]}))


def test_dates_are_parsed_day_first():
    df, _ = parsed([("05/01/2019", "3-12-19-27-41-8")])
    assert df["ds"].iloc[0] == pd.Timestamp("2019-01-05")  # 5 Jan, not 1 May


def test_last_number_is_the_superbalota():
    _, balls = parsed([("05/01/2019", "3-12-19-27-41-8")])
    assert list(balls.iloc[0]) == [3, 12, 19, 27, 41, 8]
    assert balls.shape[1] == 6


def test_clean_current_format_history_has_no_violations():
    _, balls = parsed(CLEAN)
    assert not format_violations(balls).any()
    df, _ = parsed(CLEAN)
    assert check_draw_format(df, balls) is None


@pytest.mark.parametrize(
    "ball, why",
    [
        ("3-12-19-27-44-8", "main ball above 43"),
        ("3-12-19-27-41-45", "superbalota above 16"),
        ("3-12-19-27-41-0", "superbalota below 1"),
        ("3-3-19-27-41-8", "repeated main ball"),
    ],
)
def test_format_violations_flags_impossible_draws(ball, why):
    _, balls = parsed([("05/01/2019", ball)])
    assert format_violations(balls).iloc[0], why


def test_superbalota_may_repeat_a_main_ball():
    """The five mains are drawn without replacement; the superbalota is independent."""
    _, balls = parsed([("05/01/2019", "3-12-19-27-41-12")])
    assert not format_violations(balls).any()


def test_current_format_mask_cuts_at_the_date_of_the_last_violation():
    """The era boundary is a date, not a per-row property.

    The middle row here is old-era but happens to fit today's bounds. Row-wise
    validation cannot catch it; cutting at the last violation must.
    """
    rows = [
        ("01/01/2016", "3-12-19-27-44-8"),   # provably old era (44 > 43)
        ("03/01/2016", "1-5-9-14-22-16"),    # old era, but legal under today's rules
        ("05/01/2016", "2-8-15-30-41-45"),   # provably old era (superbalota 45)
        ("07/01/2019", "3-12-19-27-41-8"),   # current era
    ]
    df, balls = parsed(rows)
    violations = format_violations(balls)
    assert list(violations) == [True, False, True, False]

    keep = current_format_mask(df, balls)
    assert list(keep) == [False, False, False, True], "the survivable old-era row must be dropped too"


def test_current_format_mask_keeps_everything_when_nothing_is_provably_old():
    df, balls = parsed(CLEAN)
    assert current_format_mask(df, balls).all()


def test_check_draw_format_reports_counts_and_the_era_boundary():
    rows = [
        ("01/01/2016", "1-5-9-14-22-16"),    # old era, legal under today's rules
        ("03/01/2016", "3-12-19-27-44-8"),   # the last provable violation
        ("07/01/2019", "3-12-19-27-41-8"),   # current era
    ]
    df, balls = parsed(rows)
    report = check_draw_format(df, balls)
    assert report["n_draws"] == 3
    assert report["n_violations"] == 1
    assert report["n_current_format"] == 1
    assert report["n_dropped_by_cutoff"] == 2
    assert report["first_violation"] == report["last_violation"] == pd.Timestamp("2016-01-03")
    assert report["current_era_starts"] == pd.Timestamp("2019-01-07")
    assert "current game cannot produce" in report["message"]


def test_current_format_mask_cannot_see_past_the_last_violation():
    """The known limit of a value-based cut, pinned so nobody mistakes it for a bug.

    The cut is the date of the *last* provable violation. An old-era draw that
    both fits today's bounds and falls after that date is indistinguishable
    from a current one and survives. Detecting it would need the real rule-change
    date, which the data does not carry.
    """
    rows = [
        ("01/01/2016", "3-12-19-27-44-8"),   # the last provable violation
        ("03/01/2016", "1-5-9-14-22-16"),    # old era, legal-looking, and later
    ]
    df, balls = parsed(rows)
    assert list(current_format_mask(df, balls)) == [False, True]


def test_mixed_eras_warn_rather_than_raise():
    """The rows are real draws; a caller may deliberately want them. Silence is the
    one option ruled out."""
    frame = draws_frame([("01/01/2016", "3-12-19-27-44-8"), ("07/01/2019", "3-12-19-27-41-8")])
    with pytest.warns(UserWarning, match="current game cannot produce"):
        preprocess_draws(frame)


def test_validate_false_stays_quiet():
    frame = draws_frame([("01/01/2016", "3-12-19-27-44-8")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        preprocess_draws(frame, validate=False)


def test_load_and_preprocess_current_format_only(tmp_path):
    path = tmp_path / "draws.csv"
    draws_frame([
        ("01/01/2016", "1-5-9-14-22-16"),
        ("03/01/2016", "3-12-19-27-44-8"),
        ("07/01/2019", "3-12-19-27-41-8"),
    ]).to_csv(path, index=False)

    df_all, balls_all = load_and_preprocess(str(path), validate=False)
    assert len(df_all) == 3

    df, balls = load_and_preprocess(str(path), validate=False, current_format_only=True)
    assert len(df) == len(balls) == 1
    assert df["ds"].iloc[0] == pd.Timestamp("2019-01-07")
    assert list(df.index) == [0] and list(balls.index) == [0], "indexes must be reset together"


def test_load_and_preprocess_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_preprocess(str(tmp_path / "nope.csv"))


def test_sample_data_is_current_format(sample):
    """The synthetic fallback must satisfy the same contract as a real export."""
    df, balls = sample
    assert not format_violations(balls).any()
    assert check_draw_format(df, balls) is None
