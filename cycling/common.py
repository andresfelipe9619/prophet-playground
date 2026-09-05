"""Result semantics and data locations for cycling. Everything else imports from here.

Two things are pinned here because getting either wrong is invisible in the
shape of a frame:

**What kind of result a row belongs to.** A rank of 4 in a stage result and a
rank of 4 in a general classification are different quantities — one is a
sprint finish, the other is three weeks of accumulated time. They are
published by the same site, in the same table layout, for the same race and
the same riders. Mixing them yields a frame where `rank` means two things,
which is this domain's version of football's opening/closing odds trap.

**Whether a rider finished.** Roughly 10-20% of a Grand Tour's start list
does not reach Paris, and abandons are not random — they concentrate among the
riders a model is least sure about. Dropping them turns "predict the finishing
order" into "predict the order among those who finished", which is a strictly
easier problem, and one nobody can bet on. So a non-finisher stays in the
frame with `rank` NaN and a `status` saying why, and filtering them out is an
explicit call to `finishers()`, never a side effect of loading.
"""

import re

# ------------------------------------------------------------------ the kinds

STAGE = "stage"          # one stage of a stage race: rank on the day
ONE_DAY = "one_day"      # a single-day race: rank at the line
GC = "gc"                # general classification: rank on accumulated time

RESULT_KINDS = (STAGE, ONE_DAY, GC)

RESULT_KIND_LABELS = {
    STAGE: "Stage result",
    ONE_DAY: "One-day race result",
    GC: "General classification",
}

# The kinds whose `rank` is a placing on a single day's racing. GC is excluded
# on purpose: it is the one whose rank aggregates several days, and code that
# treats a day's result and a standing alike is the bug this tuple exists for.
SINGLE_DAY_KINDS = (STAGE, ONE_DAY)

# ---------------------------------------------------------------- the statuses

FINISHED = "FIN"
DNF = "DNF"    # did not finish — abandoned
DNS = "DNS"    # did not start
DSQ = "DSQ"    # disqualified
OTL = "OTL"    # outside the time limit: finished, but eliminated
NR = "NR"      # not ranked, reason unpublished

STATUSES = (FINISHED, DNF, DNS, DSQ, OTL, NR)
NON_FINISHER_STATUSES = tuple(s for s in STATUSES if s != FINISHED)

STATUS_LABELS = {
    FINISHED: "Finished",
    DNF: "Did not finish",
    DNS: "Did not start",
    DSQ: "Disqualified",
    OTL: "Outside the time limit",
    NR: "Not ranked",
}

# How a results page spells each of them. Keyed on the upper-cased cell text
# with punctuation stripped, so 'DNF', 'dnf' and 'D.N.F.' all resolve.
STATUS_ALIASES = {
    "DNF": DNF, "AB": DNF, "ABD": DNF, "ABANDON": DNF,
    "DNS": DNS, "NS": DNS,
    "DSQ": DSQ, "DQ": DSQ, "DF": DSQ,
    "OTL": OTL, "HD": OTL, "TL": OTL,
    "NR": NR, "": NR, "-": NR,
}

DEFAULT_DATA_DIR = "exported_data/cycling"

# The tidy shape every loader produces.
RESULT_COLUMNS = ["ds", "race", "kind", "stage", "rank", "rider", "team",
                  "status", "time_seconds"]

# The on-disk column names the scraper writes and the processor reads. Kept
# distinct from RESULT_COLUMNS so the file format and the in-memory shape can
# be told apart in a traceback.
CSV_COLUMNS = ["Date", "Race", "Kind", "Stage", "Rank", "Rider", "Team",
               "Status", "TimeSeconds"]


def is_finisher(status):
    """Did this rider produce a ranked finish?

    `OTL` is deliberately False: a rider outside the time limit crossed the
    line but is removed from the classification, so they have no rank and
    counting them as a finisher would leave a hole where a rank should be.
    """
    return status == FINISHED


def finishers(results):
    """The ranked finishers only — an explicit filter, never applied on load."""
    return results[results["status"] == FINISHED]


def normalise_status(text):
    """A results-page rank cell -> a status, or None when it is a placing.

    Returns None for '1', '12', '3.' and the like: those are ranks, and the
    caller reads the number. Anything non-numeric must map to a known status;
    an unrecognised marker raises rather than becoming a silent 'NR', because
    an unknown abbreviation is exactly how a page whose markup changed slips
    through looking plausible.
    """
    cleaned = re.sub(r"[^A-Za-z0-9-]", "", str(text)).upper()
    if re.fullmatch(r"\d+", cleaned):
        return None
    status = STATUS_ALIASES.get(cleaned)
    if status is None:
        raise ValueError(
            f"Unknown rank marker {text!r}. Known non-finisher markers: "
            f"{', '.join(sorted(k for k in STATUS_ALIASES if k))}. Add it to STATUS_ALIASES "
            "only after checking on the page what it means — guessing loses riders."
        )
    return status


def parse_time_to_seconds(text):
    """'4:15:22' -> 15322.0, '0:14' -> 14.0, '22' -> 22.0. None when blank.

    Cycling times come as h:mm:ss, mm:ss or bare seconds, with an optional
    leading '+' on a gap. Returns a float count of seconds and says nothing
    about whether that is a total or a gap — resolving that is the scraper's
    job, and `time_seconds` in the contract is **always a total**.
    """
    if text is None:
        return None
    cleaned = str(text).strip().replace("+", "").replace("''", "").strip()
    if not cleaned or cleaned in {"-", ",,", ",,,"}:
        return None

    parts = cleaned.split(":")
    if len(parts) > 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"Could not read {text!r} as a cycling time (h:mm:ss, mm:ss or ss).")

    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def format_seconds(seconds):
    """15322.0 -> '4:15:22', for printing a scrape before trusting it."""
    if seconds is None:
        return ""
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
