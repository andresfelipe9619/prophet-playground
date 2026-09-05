"""Cycling result semantics — cycling/common.py.

The two things pinned here are the ones no range check can catch: what counts
as a finish (`OTL` does not), and that an unknown rank marker raises instead of
becoming a silent 'not ranked'. A results page that starts spelling abandons
differently would otherwise turn a third of the field into NR rows and look
entirely plausible.
"""

import pytest

from cycling.common import (
    CSV_COLUMNS,
    DNF,
    DNS,
    DSQ,
    FINISHED,
    GC,
    NON_FINISHER_STATUSES,
    NR,
    ONE_DAY,
    OTL,
    RESULT_COLUMNS,
    RESULT_KINDS,
    SINGLE_DAY_KINDS,
    STAGE,
    STATUSES,
    format_seconds,
    is_finisher,
    normalise_status,
    parse_time_to_seconds,
)


def test_the_kinds_and_statuses_are_pinned():
    assert RESULT_KINDS == (STAGE, ONE_DAY, GC)
    assert STATUSES == (FINISHED, DNF, DNS, DSQ, OTL, NR)
    assert set(NON_FINISHER_STATUSES) == set(STATUSES) - {FINISHED}


def test_the_general_classification_is_not_a_single_day_result():
    # Its rank aggregates three weeks; code that treats it as a day's placing
    # is the bug SINGLE_DAY_KINDS exists to prevent.
    assert GC not in SINGLE_DAY_KINDS
    assert set(SINGLE_DAY_KINDS) == {STAGE, ONE_DAY}


def test_outside_the_time_limit_is_not_a_finisher():
    # The rider crossed the line but is removed from the classification, so
    # they have no rank; counting them as a finisher leaves a hole where one
    # should be.
    assert is_finisher(FINISHED)
    assert not is_finisher(OTL)
    assert not any(is_finisher(s) for s in NON_FINISHER_STATUSES)


def test_the_csv_and_tidy_shapes_stay_distinct_but_aligned():
    assert len(CSV_COLUMNS) == len(RESULT_COLUMNS)
    assert [c.lower() for c in CSV_COLUMNS][1:4] == ["race", "kind", "stage"]


# ----------------------------------------------------------------- statuses

def test_a_placing_is_not_a_status():
    assert normalise_status("1") is None
    assert normalise_status("12.") is None


def test_the_published_abandon_markers_resolve():
    assert normalise_status("DNF") == DNF
    assert normalise_status("dnf") == DNF
    assert normalise_status("D.N.F.") == DNF
    assert normalise_status("DNS") == DNS
    assert normalise_status("DSQ") == DSQ
    assert normalise_status("OTL") == OTL
    assert normalise_status("") == NR


def test_an_unknown_marker_raises_rather_than_becoming_not_ranked():
    with pytest.raises(ValueError, match="Unknown rank marker"):
        normalise_status("HORS")


# -------------------------------------------------------------------- times

@pytest.mark.parametrize("text,seconds", [
    ("4:15:22", 15322.0),
    ("0:14", 14.0),
    ("14", 14.0),
    ("+0:14", 14.0),
    ("1:00:00", 3600.0),
])
def test_cycling_times_parse_in_every_published_form(text, seconds):
    assert parse_time_to_seconds(text) == seconds


@pytest.mark.parametrize("text", ["", "   ", "-", ",,", None])
def test_a_blank_time_is_none_not_zero(text):
    # Zero would make an unknown time the fastest in the race.
    assert parse_time_to_seconds(text) is None


def test_an_unreadable_time_raises():
    with pytest.raises(ValueError, match="Could not read"):
        parse_time_to_seconds("4h15")


def test_seconds_format_back_for_eyeballing_a_scrape():
    assert format_seconds(15322) == "4:15:22"
    assert format_seconds(parse_time_to_seconds("0:14")) == "0:00:14"
