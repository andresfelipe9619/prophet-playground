"""Forward records for football and cycling — the domains where one would mean something.

`tests/test_core_registry.py` pins the three refusals. These pin what each
domain adds on top: that a football row is scored **against the closing price**
rather than for being right, and that a cycling row is scored over **the field
it was registered against** rather than over whoever turned up.
"""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.registry import RegistryError
from cycling import registry as cy
from cycling.baseline import form_worths
from cycling.processor import preprocess_results
from cycling.sample_data import generate_stage_race
from football import registry as fb
from football.common import OUTCOMES, PROBABILITY_COLUMNS
from football.market import market_probabilities
from football.processor import preprocess_matches
from football.sample_data import generate_matches

FUTURE = (datetime.now(UTC) + timedelta(days=3)).date().isoformat()
PAST = (datetime.now(UTC) - timedelta(days=3)).date().isoformat()


# ===================================================================== football


@pytest.fixture
def fb_path(tmp_path):
    return str(tmp_path / "football_predictions.csv")


@pytest.fixture
def played():
    """A played season with the de-margined market beside every result."""
    raw = generate_matches(n_teams=10, seed=5, market_noise=0.7).drop(
        columns=["TrueH", "TrueD", "TrueA"])
    return market_probabilities(preprocess_matches(raw, validate=False))


def _fb_record(path, probabilities=(0.5, 0.3, 0.2), home="A", away="B", label="model", **kw):
    return fb.record(probabilities, FUTURE, home, away, label, path=path, **kw)


def test_a_football_forecast_must_be_a_probability_vector(fb_path):
    with pytest.raises(RegistryError, match="3 probabilities"):
        _fb_record(fb_path, probabilities=(0.5, 0.5))


def test_an_unnormalised_forecast_is_refused(fb_path):
    """Which of the two readings was meant is not recoverable after the fact."""
    with pytest.raises(RegistryError, match="sum to"):
        _fb_record(fb_path, probabilities=(0.5, 0.3, 0.3))


def test_a_negative_probability_is_refused(fb_path):
    with pytest.raises(RegistryError, match="finite and non-negative"):
        _fb_record(fb_path, probabilities=(1.2, -0.1, -0.1))


def test_a_fixture_already_played_is_refused(fb_path):
    with pytest.raises(RegistryError, match="not in the future"):
        fb.record((0.5, 0.3, 0.2), PAST, "A", "B", "model", path=fb_path)


def test_the_forecast_is_stored_in_outcomes_order(fb_path):
    _fb_record(fb_path, probabilities=(0.5, 0.3, 0.2))
    row = fb.load(fb_path).iloc[0]
    assert [row[c] for c in PROBABILITY_COLUMNS] == pytest.approx([0.5, 0.3, 0.2])


def test_a_forecast_is_scored_against_the_closing_price_not_for_being_right(fb_path, played):
    """The whole point. A forecast that backs the favourite every week is right
    about half the time and has no edge, so 'right' is not what is recorded."""
    # A priced fixture: one with no market vector is left pending by design,
    # which the next test is about.
    match = played.dropna(subset=list(PROBABILITY_COLUMNS)).iloc[0]
    fb.record((0.5, 0.3, 0.2), FUTURE, match["home_team"], match["away_team"],
              "model", path=fb_path)

    # Move the fixture into the past so it can be scored, which is only
    # expressible by editing the file — the refusal is doing its job.
    _age(fb_path, "match_date", match["ds"])
    registry = fb.score_pending(played, path=fb_path)

    row = registry.iloc[0]
    assert row["outcome"] == match["outcome"]
    # score_difference is market - model, so a model that scored better than
    # the price leaves a positive one. Same sign convention as evaluation.py.
    assert row["score_difference"] == pytest.approx(row["market_score"] - row["model_score"])


def test_a_fixture_with_no_market_price_is_left_pending(fb_path, played):
    """A forecast with no bar beside it is a bare model number, which is the
    thing every football surface here refuses to show."""
    priced = played.dropna(subset=list(PROBABILITY_COLUMNS))
    match = priced.iloc[0]
    fb.record((0.5, 0.3, 0.2), FUTURE, match["home_team"], match["away_team"],
              "model", path=fb_path)
    _age(fb_path, "match_date", match["ds"])

    blinded = played.copy()
    blinded.loc[match.name, list(PROBABILITY_COLUMNS)] = np.nan
    registry = fb.score_pending(blinded, path=fb_path)
    assert registry["scored_at"].isna().all()


def test_scoring_refuses_a_frame_with_no_market_probabilities(fb_path, played):
    _fb_record(fb_path)
    with pytest.raises(ValueError, match="no market probabilities"):
        fb.score_pending(played.drop(columns=list(PROBABILITY_COLUMNS)), path=fb_path)


def test_a_perfect_forecast_beats_the_market_on_the_record(fb_path, played):
    """The positive control: a forecast that knew every result must come out
    ahead of the price, or the accumulated difference means nothing."""
    subset = played.dropna(subset=list(PROBABILITY_COLUMNS)).head(40)
    column = {outcome: i for i, outcome in enumerate(OUTCOMES)}
    for match in subset.itertuples():
        vector = [0.02, 0.02, 0.02]
        vector[column[match.outcome]] = 0.96
        fb.record(vector, FUTURE, match.home_team, match.away_team,
                  f"{match.home_team}-{match.away_team}", path=fb_path)
    _age_fixtures(fb_path, subset)

    fb.score_pending(subset, path=fb_path)
    table = fb.summary(fb_path)
    assert table["effect"].iloc[0] > 0
    assert bool(table["beats_market_corrected"].iloc[0]) is True


def test_a_forecast_that_is_the_market_has_no_edge_on_the_record(fb_path, played):
    """The endpoint, the same one ensemble.py's weight-0 blend pins: registering
    the price itself must accumulate a difference of exactly zero."""
    subset = played.dropna(subset=list(PROBABILITY_COLUMNS)).head(40)
    for match in subset.itertuples():
        vector = [getattr(match, c) for c in PROBABILITY_COLUMNS]
        fb.record(vector, FUTURE, match.home_team, match.away_team,
                  f"{match.home_team}-{match.away_team}", path=fb_path)
    _age_fixtures(fb_path, subset)

    registry = fb.score_pending(subset, path=fb_path)
    np.testing.assert_allclose(registry["score_difference"].astype(float), 0.0, atol=1e-12)
    assert bool(fb.summary(fb_path)["beats_market_corrected"].iloc[0]) is False


def test_the_football_summary_of_an_empty_registry_has_the_agreed_columns(fb_path):
    table = fb.summary(fb_path)
    assert table.empty
    assert "beats_market_corrected" in table.columns


def test_the_correction_tightens_with_more_labels(fb_path, played):
    subset = played.dropna(subset=list(PROBABILITY_COLUMNS)).head(20)
    for match in subset.itertuples():
        for label in ("one", "two"):
            fb.record([0.4, 0.3, 0.3], FUTURE, match.home_team, match.away_team,
                      f"{label}-{match.home_team}-{match.away_team}", path=fb_path)
    _age_fixtures(fb_path, subset)
    fb.score_pending(subset, path=fb_path)

    pooled = fb.summary(fb_path)
    split = fb.summary(fb_path, by_label=True)
    assert split["bonferroni_threshold"].iloc[0] < pooled["bonferroni_threshold"].iloc[0]


def _age(path, column, when):
    """Backdate every registered event to one date.

    Only a test does this, and only through the file: `record` cannot write a
    past event, which is exactly the refusal being relied on everywhere else.
    Use it when the registry holds one event; for several, use `_age_fixtures`,
    which gives each row the date of the fixture it actually names.
    """
    frame = pd.read_csv(path)
    frame[column] = pd.Timestamp(when).strftime("%Y-%m-%d")
    frame.to_csv(path, index=False)


def _age_fixtures(path, matches):
    """Backdate each registered fixture to its own kick-off.

    One flat date for everything would leave every row but that day's unable to
    resolve, which is how the first version of these tests failed: not a defect
    in the registry, a defect in the harness.
    """
    when = {(m.home_team, m.away_team): pd.Timestamp(m.ds).strftime("%Y-%m-%d")
            for m in matches.itertuples()}
    frame = pd.read_csv(path)
    frame["match_date"] = [when[(home, away)] for home, away
                           in zip(frame["home_team"], frame["away_team"], strict=True)]
    frame.to_csv(path, index=False)


# ====================================================================== cycling


@pytest.fixture
def cy_path(tmp_path):
    return str(tmp_path / "cycling_predictions.csv")


@pytest.fixture
def race():
    raw = generate_stage_race(n_riders=18, n_stages=4, seed=6,
                              climbing_stages={1, 2, 3, 4})
    return preprocess_results(raw, validate=False)


def test_a_cycling_forecast_must_line_up_with_its_start_list(cy_path):
    with pytest.raises(RegistryError, match="riders against"):
        cy.record(["a", "b"], [1.0], FUTURE, "tour", "model", path=cy_path)


def test_a_zero_worth_is_refused(cy_path):
    """It takes a logarithm to minus infinity in the Plackett-Luce score."""
    with pytest.raises(RegistryError, match="finite and strictly positive"):
        cy.record(["a", "b"], [1.0, 0.0], FUTURE, "tour", "model", path=cy_path)


def test_a_duplicated_rider_is_refused(cy_path):
    with pytest.raises(RegistryError, match="twice"):
        cy.record(["a", "a"], [1.0, 1.0], FUTURE, "tour", "model", path=cy_path)


def test_the_start_list_survives_a_reload(cy_path):
    cy.record(["a", "b"], [2.0, 1.0], FUTURE, "tour", "model", path=cy_path)
    row = cy.load(cy_path).iloc[0]
    assert json.loads(row["riders"]) == ["a", "b"]
    assert json.loads(row["worths"]) == [2.0, 1.0]


def test_a_cycling_forecast_is_scored_against_the_ranking(cy_path, race):
    stage = race[race["stage"] == race["stage"].max()]
    riders = list(stage["rider"])
    cy.record(riders, np.ones(len(riders)), FUTURE, stage["race"].iloc[0], "uniform",
              kind=stage["kind"].iloc[0], stage=int(stage["stage"].iloc[0]), path=cy_path)
    _age(cy_path, "race_date", stage["ds"].min())

    registry = cy.score_pending(race, form_worths, path=cy_path)
    row = registry.iloc[0]
    assert row["metric"] == "plackett_luce"
    assert row["score_difference"] == pytest.approx(row["baseline_score"] - row["model_score"])


def test_a_forecast_that_is_the_ranking_scores_exactly_zero(cy_path, race):
    """The endpoint. A registered forecast identical to its own baseline must
    accumulate nothing, or 'beats the ranking' stops meaning anything."""
    stage = race[race["stage"] == race["stage"].max()]
    riders = list(stage["rider"])
    as_of = stage["ds"].min()
    baseline = form_worths(race[race["ds"] < as_of], riders, as_of=as_of)

    cy.record(riders, baseline, FUTURE, stage["race"].iloc[0], "is-the-ranking",
              kind=stage["kind"].iloc[0], stage=int(stage["stage"].iloc[0]), path=cy_path)
    _age(cy_path, "race_date", as_of)

    registry = cy.score_pending(race, form_worths, path=cy_path)
    assert float(registry["score_difference"].iloc[0]) == pytest.approx(0.0, abs=1e-12)


def test_scoring_refuses_a_race_whose_field_is_not_what_was_predicted(cy_path, race):
    """The registry's version of the rule the scoring already enforces: shrinking
    the field turns the claim into the strictly easier one."""
    stage = race[race["stage"] == race["stage"].max()]
    riders = list(stage["rider"])
    cy.record(riders, np.ones(len(riders)), FUTURE, stage["race"].iloc[0], "uniform",
              kind=stage["kind"].iloc[0], stage=int(stage["stage"].iloc[0]), path=cy_path)
    _age(cy_path, "race_date", stage["ds"].min())

    thinned = race.drop(index=stage.index[:3])
    with pytest.raises(RegistryError, match="different, easier claim"):
        cy.score_pending(thinned, form_worths, path=cy_path)


def test_a_race_not_in_the_frame_is_left_pending(cy_path, race):
    cy.record(["x", "y"], [1.0, 2.0], FUTURE, "a-race-that-did-not-happen", "model",
              path=cy_path)
    registry = cy.score_pending(race, form_worths, path=cy_path)
    assert registry["scored_at"].isna().all()


def test_the_cycling_summary_of_an_empty_registry_has_the_agreed_columns(cy_path):
    table = cy.summary(cy_path)
    assert table.empty
    assert "beats_baseline_corrected" in table.columns


def test_cycling_status_names_the_next_race(cy_path):
    cy.record(["a", "b"], [1.0, 2.0], FUTURE, "tour", "model", path=cy_path)
    state = cy.status(cy_path)
    assert state["n_pending"] == 1
    assert state["next_race"] == pd.Timestamp(FUTURE)


def test_a_cycling_race_still_to_come_is_listed_as_pending(cy_path, race):
    cy.record(["a", "b"], [1.0, 2.0], FUTURE, "tour", "model", path=cy_path)

    assert list(cy.pending(cy_path)["race"]) == ["tour"]


def _age_stages(path, race):
    """Backdate each registered row to the date of the stage it names.

    The cycling counterpart of `_age_fixtures`, and for the same reason: one
    flat date across several stages leaves every row but that day's unable to
    resolve, which reads in the results as a scorer that skipped them.
    """
    when = {int(stage): group["ds"].min().strftime("%Y-%m-%d")
            for stage, group in race.groupby("stage")}
    frame = pd.read_csv(path)
    frame["race_date"] = [when[int(stage)] for stage in frame["stage"]]
    frame.to_csv(path, index=False)


def _register_stages(cy_path, race, worths_for, label="model", stages=(2, 3, 4)):
    """Register one forecast per stage, from `worths_for(riders, as_of, history)`."""
    for stage_number in stages:
        stage = race[race["stage"] == stage_number]
        riders = list(stage["rider"])
        as_of = stage["ds"].min()
        # Each stage needs its own future date: (event, label) is unique by the
        # registry's second refusal, so one flat FUTURE would read as a second
        # prediction for the same race rather than a prediction for the next.
        race_date = (datetime.now(UTC) + timedelta(days=3 + stage_number)).date().isoformat()
        cy.record(riders, worths_for(riders, as_of, race[race["ds"] < as_of]),
                  race_date, stage["race"].iloc[0], label, kind=stage["kind"].iloc[0],
                  stage=stage_number, path=cy_path)
    _age_stages(cy_path, race)


def test_the_cycling_summary_tests_the_scored_races_against_the_ranking(cy_path, race):
    _register_stages(cy_path, race, lambda riders, as_of, history: np.ones(len(riders)))
    cy.score_pending(race, form_worths, path=cy_path)

    table = cy.summary(cy_path)

    # Pooled by default: one row covering every scored race, labelled "all".
    assert list(table["label"]) == ["all"]
    assert int(table["n_scored"].iloc[0]) == 3
    # One label, so the corrected threshold is alpha itself -- the tightening
    # only starts when a second challenger is registered against the same races.
    assert float(table["bonferroni_threshold"].iloc[0]) == pytest.approx(0.05)
    assert bool(table["beats_baseline_corrected"].iloc[0]) in (True, False)


def test_a_registered_forecast_that_is_the_ranking_accumulates_exactly_zero(cy_path, race):
    """The endpoint again, now through `summary` rather than one row.

    Every race scores identically to its own baseline, so the two mean scores
    come out equal — and because the differences are all exactly zero they have
    no spread, which the z-test reports as no estimable effect rather than as a
    very precise zero. Both halves matter: the accumulation is nil, and the
    summary does not turn a degenerate sample into a verdict.
    """
    _register_stages(cy_path, race,
                     lambda riders, as_of, history: form_worths(history, riders, as_of=as_of))
    registry = cy.score_pending(race, form_worths, path=cy_path)

    assert registry["score_difference"].abs().max() == pytest.approx(0.0, abs=1e-12)

    table = cy.summary(cy_path)

    assert float(table["model_score"].iloc[0]) == pytest.approx(float(table["baseline_score"].iloc[0]))
    assert pd.isna(table["effect"].iloc[0])
    assert not bool(table["beats_baseline"].iloc[0])
    assert not bool(table["beats_baseline_corrected"].iloc[0])


def test_a_second_label_tightens_the_corrected_threshold(cy_path, race):
    """`n_comparisons` is the label count, so registering another challenger
    against the same races makes the bar harder — the correct direction."""
    _register_stages(cy_path, race, lambda riders, as_of, history: np.ones(len(riders)),
                     label="uniform")
    _register_stages(cy_path, race,
                     lambda riders, as_of, history: form_worths(history, riders, as_of=as_of),
                     label="is-the-ranking")
    _age_stages(cy_path, race)
    cy.score_pending(race, form_worths, path=cy_path)

    table = cy.summary(cy_path, by_label=True)

    assert sorted(table["label"]) == ["is-the-ranking", "uniform"]
    assert set(table["bonferroni_threshold"]) == {0.05 / 2}


def test_a_single_scored_race_reports_no_interval_rather_than_a_fabricated_one(cy_path, race):
    """One observation has no spread, so the effect is not estimable. Saying so
    beats returning a z-test built on a variance of nothing."""
    _register_stages(cy_path, race, lambda riders, as_of, history: np.ones(len(riders)),
                     stages=(3,))
    cy.score_pending(race, form_worths, path=cy_path)

    table = cy.summary(cy_path)

    assert int(table["n_scored"].iloc[0]) == 1
    assert pd.isna(table["p_value_greater"].iloc[0])
    assert not bool(table["beats_baseline_corrected"].iloc[0])
