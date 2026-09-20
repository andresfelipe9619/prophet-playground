"""Did the price move toward the bet? — football/clv.py

Two controls hold this file up.

The **endpoint**: CLV measured against the very price you bet is exactly zero,
the same load-bearing zero as `ensemble.py`'s weight-0 blend. Everything else
here is read as a departure from it, so if it drifts nothing below means
anything.

The **positive control**: a bettor who takes a soft opening price against a
closing line that knows the truth must show positive CLV. It is built from
`football/sample_data.py`'s `p_true_*`, the generator's answer key, which
nothing outside tests may read — this is a test.
"""

import numpy as np
import pandas as pd
import pytest

from football.clv import (
    BET_ODDS_COLUMNS,
    CLOSING_ODDS_COLUMNS,
    PriceJoinError,
    beats_closing_test,
    clv,
    clv_report,
    clv_table,
    join_prices,
    paired_prices,
    summarise_by_outcome,
)
from football.common import ODDS_COLUMNS, OUTCOMES
from football.market import fair_odds
from football.processor import preprocess_matches
from football.sample_data import generate_matches


def _matches(seed=1, n_teams=14, market_noise=0.8):
    raw = generate_matches(n_teams=n_teams, seed=seed, market_noise=market_noise).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    return preprocess_matches(raw, validate=False)


def _truth_priced(seed=1, n_teams=14, margin=0.05):
    """An opening frame with soft prices and a closing frame that prices the truth.

    `p_true_*` is the generative answer key. A closing line built from it is
    the sharpest book that can exist, so a bettor who found value against the
    soft opening prices must be shown to have had it.
    """
    raw = generate_matches(n_teams=n_teams, seed=seed, market_noise=1.4)
    truth = raw[["TrueH", "TrueD", "TrueA"]].to_numpy()
    opening = preprocess_matches(raw.drop(columns=["TrueH", "TrueD", "TrueA"]), validate=False)

    closing = opening.copy()
    priced = fair_odds(truth / truth.sum(axis=1, keepdims=True) * (1.0 + margin))
    for column, values in zip(ODDS_COLUMNS, priced.T, strict=True):
        closing[column] = values
    return opening, closing, truth


def _priced_frame(beliefs, overround):
    """Two matches priced from given beliefs under a given margin."""
    odds = fair_odds(np.asarray(beliefs, dtype=float) * overround)
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2024-08-10", "2024-08-11"])[:len(odds)],
        "home_team": ["A", "C"][:len(odds)],
        "away_team": ["B", "D"][:len(odds)],
        "outcome": ["H", "A"][:len(odds)],
    })
    for column, values in zip(ODDS_COLUMNS, odds.T, strict=True):
        frame[column] = values
    return frame


# ------------------------------------------------------------------ the join


def test_the_join_pairs_a_match_with_itself(frame=None):
    matches = _matches()
    joined = join_prices(matches, matches)
    assert len(joined) == len(matches)
    assert set(BET_ODDS_COLUMNS) <= set(joined.columns)
    assert set(CLOSING_ODDS_COLUMNS) <= set(joined.columns)
    assert joined.attrs["n_dropped"] == 0


def test_the_join_reports_what_it_dropped():
    """Silently narrowing the sample would redefine what any verdict is about."""
    matches = _matches()
    joined = join_prices(matches, matches.iloc[:-20])
    assert len(joined) == len(matches) - 20
    assert joined.attrs["n_dropped"] == 20


def test_the_join_refuses_frames_with_nothing_in_common():
    """An empty CLV table and a CLV of zero look identical downstream and mean
    opposite things, so this raises rather than returning one."""
    matches = _matches()
    elsewhere = matches.copy()
    elsewhere["ds"] = elsewhere["ds"] + pd.Timedelta(days=4000)

    with pytest.raises(PriceJoinError, match="No match appears in both"):
        join_prices(matches, elsewhere)


def test_the_join_names_a_missing_column_rather_than_failing_later():
    matches = _matches()
    with pytest.raises(PriceJoinError, match="odds_home"):
        join_prices(matches.drop(columns=["odds_home"]), matches)


def test_the_join_keeps_both_odds_sources():
    """Which side is the closing one is the whole premise; it has to survive."""
    matches = _matches()
    joined = join_prices(matches, matches)
    assert joined.attrs["closing_odds_source"] == matches.attrs.get("odds_source")


# --------------------------------------------------------------- the endpoint


def test_clv_against_the_price_you_bet_is_exactly_zero():
    """The load-bearing zero. Every other number here is read as a departure
    from it, the same role weight 0 plays in ensemble.py."""
    matches = _matches()
    table, result = clv_report(join_prices(matches, matches))

    priced = table["clv"].notna()
    assert priced.any()
    np.testing.assert_allclose(table.loc[priced, "clv"], 0.0, atol=1e-12)
    assert result["mean_clv"] == pytest.approx(0.0, abs=1e-12)
    assert result["beats_closing_corrected"] is False


def test_a_match_with_no_usable_price_stays_in_the_table_as_nan():
    """Alignment over convenience, the rule market.py already follows: the row
    is still a match, and dropping it here would make the CLV table a different
    length from the frame it came from. The test drops them, not the table."""
    matches = _matches()
    table, result = clv_report(join_prices(matches, matches))

    assert len(table) == len(matches)
    assert table["clv"].isna().any(), "fixture had complete prices on every row"
    assert result["n_bets"] == int(table["clv"].notna().sum())


def test_a_wider_margin_at_the_close_is_not_a_loss_of_value():
    """The reason CLV is measured de-margined.

    A book that widens its margin moves every raw price against every bettor.
    A raw `1/odds` version reads that as everyone losing value; it is a fact
    about the book, not about the bet.
    """
    # A hand-built pair, because the point is exact: the same de-margined
    # beliefs priced under two different overrounds. Scaling a generated frame's
    # prices instead can push a short favourite past 1.0, which is not a price
    # at all and which market.py rightly refuses.
    beliefs = np.array([[0.50, 0.30, 0.20], [0.40, 0.28, 0.32]])
    thin, fat = _priced_frame(beliefs, 1.03), _priced_frame(beliefs, 1.12)

    table, result = clv_report(join_prices(thin, fat), bets=["H", "A"])
    assert result["mean_clv"] == pytest.approx(0.0, abs=1e-9)
    assert len(table) == len(thin)

    raw_shift = (1.0 / fat["odds_home"] - 1.0 / thin["odds_home"]).mean()
    assert raw_shift > 0.005, "fixture did not actually widen the margin"


# --------------------------------------------------------- the positive control


def test_a_bettor_who_found_real_value_shows_positive_clv():
    opening, closing, truth = _truth_priced()
    # Back the outcome the soft opening price under-rates most against the truth.
    backed = [OUTCOMES[i] for i in truth.argmax(axis=1)]

    _, result = clv_report(join_prices(opening, closing), bets=backed)
    assert result["mean_clv"] > 0
    assert result["beats_closing_corrected"] is True
    assert result["hit_rate"] > 0.5


def test_a_bettor_on_the_wrong_side_does_not_pass_the_one_sided_test():
    """Systematically *behind* the market's revision is the worst result there
    is, and a two-sided reading would present it as the best-looking one."""
    opening, closing, truth = _truth_priced()
    backed = [OUTCOMES[i] for i in truth.argmin(axis=1)]

    _, result = clv_report(join_prices(opening, closing), bets=backed)
    assert result["mean_clv"] < 0
    assert result["beats_closing"] is False
    assert result["beats_closing_corrected"] is False


# ------------------------------------------------------------------- the test


def test_the_correction_tightens_as_more_is_measured_at_once():
    matches = _matches()
    values = np.full(200, 0.004)
    one = beats_closing_test(values, n_comparisons=1)
    five = beats_closing_test(values, n_comparisons=5)
    assert five["bonferroni_threshold"] == pytest.approx(one["bonferroni_threshold"] / 5)
    assert len(matches)  # fixture touched, so a change to it fails here too


def test_too_few_bets_is_not_a_pass():
    result = beats_closing_test([0.5])
    assert result["beats_closing"] is False
    assert result["beats_closing_corrected"] is False
    assert np.isnan(result["mean_clv"])


def test_the_hit_rate_and_the_mean_are_reported_together():
    """They answer different questions. A hit rate near a half with a good mean
    is a handful of large wins, which is a different claim about a bettor."""
    lopsided = np.array([-0.001] * 90 + [0.05] * 10)
    result = beats_closing_test(lopsided)
    assert result["hit_rate"] == pytest.approx(0.10)
    assert result["mean_clv"] > 0


def test_the_interval_travels_with_the_verdict():
    result = beats_closing_test(np.full(300, 0.01) + np.linspace(-0.01, 0.01, 300))
    assert result["ci_low"] < result["effect"] < result["ci_high"]


# -------------------------------------------------------------- bookkeeping


def test_clv_refuses_a_mismatched_number_of_bets():
    matches = _matches()
    joined = join_prices(matches, matches)
    with pytest.raises(ValueError, match="prices against"):
        clv(joined[list(BET_ODDS_COLUMNS)].to_numpy(),
            joined[list(CLOSING_ODDS_COLUMNS)].to_numpy(),
            list(joined["outcome"])[:-1])


def test_the_table_carries_one_row_per_bet_and_names_what_was_backed():
    matches = _matches()
    table = clv_table(join_prices(matches, matches))
    assert len(table) == len(matches)
    assert set(table["bet"]) <= set(OUTCOMES)
    assert list(table.columns[:3]) == ["ds", "home_team", "away_team"]


def test_the_per_outcome_split_is_a_diagnostic_with_every_row_present():
    """Every outcome gets a row even at zero bets: a missing row reads as an
    oversight, and reading the best of three slices is the mistake this
    repository corrects everywhere else."""
    opening, closing, truth = _truth_priced()
    backed = [OUTCOMES[i] for i in truth.argmax(axis=1)]
    table = clv_table(join_prices(opening, closing), bets=backed)

    split = summarise_by_outcome(table)
    assert list(split["bet"]) == list(OUTCOMES)
    assert split["n_bets"].sum() == len(table)


def test_the_report_says_which_side_was_which():
    """A CLV computed against opening prices on both ends is not CLV at all."""
    opening, closing, _ = _truth_priced()
    _, result = clv_report(join_prices(opening, closing))
    assert "bet_odds_source" in result and "closing_odds_source" in result
    assert result["n_dropped"] == 0


# ------------------------------------- both ends of the line from one file


def _two_priced(seed=2, n_teams=14, market_noise=0.15, opening_noise=1.2):
    """A file shaped like a real 2019/20-or-later season: two prices per match."""
    return generate_matches(n_teams=n_teams, seed=seed, market_noise=market_noise,
                            opening_noise=opening_noise)


def test_one_season_file_yields_both_ends_of_the_line():
    """From 2019/20 the opening and closing columns sit side by side, which is
    what makes CLV measurable here without a second data source."""
    joined = paired_prices(_two_priced())

    assert joined.attrs["bet_odds_source"] == "market_opening_average"
    assert joined.attrs["closing_odds_source"] == "market_closing_average"
    assert joined.attrs["n_dropped"] == 0


def test_the_pairing_goes_through_the_contract_twice_rather_than_around_it():
    """Neither resolved frame may carry both families. The guard that makes
    that true is still processor.py's, not a second copy of it here."""
    raw = _two_priced()
    joined = paired_prices(raw)

    opening = preprocess_matches(raw.drop(columns=["AvgCH", "AvgCD", "AvgCA"]), validate=False)
    closing = preprocess_matches(raw, validate=False)
    np.testing.assert_allclose(joined[list(BET_ODDS_COLUMNS)].to_numpy(),
                               opening[list(ODDS_COLUMNS)].to_numpy())
    np.testing.assert_allclose(joined[list(CLOSING_ODDS_COLUMNS)].to_numpy(),
                               closing[list(ODDS_COLUMNS)].to_numpy())


def test_a_file_with_no_closing_prices_is_refused_by_name():
    """A pre-2019 season: the best available is an opening line, and CLV
    measured against opening prices on both ends is not CLV."""
    raw = generate_matches(n_teams=10, seed=3, closing_odds=False)
    with pytest.raises(PriceJoinError, match="opening line"):
        paired_prices(raw)


def test_a_sharpening_line_is_what_makes_clv_measurable():
    """The two blurs are independent draws from one truth. If the generator
    ever reused a single blur, every CLV would be exactly zero and every test
    in this section would pass while measuring nothing."""
    raw = _two_priced()
    joined = paired_prices(raw)
    table = clv_table(joined)

    moved = table["clv"].dropna()
    assert (moved != 0).mean() > 0.95, "opening and closing prices are the same number"
