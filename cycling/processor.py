"""Owner of the cycling data contract: scraped result rows in, tidy results out.

This is the cycling counterpart of `lottery/utils/processor.py` and
`football/processor.py`: the single place that knows what a result file looks
like, so the format cannot drift between the scraper, a hand-edited CSV and
the synthetic generator.

**The three decisions it exists to protect.**

*One kind of result per frame.* A rank in a stage result and a rank in a
general classification are different quantities measured over different
periods, published in the same table layout for the same riders. A frame
holding both has a `rank` column that means two things, and every model
fitted on it is fitted on a mixture. So a mixed file raises, and `load_races`
raises rather than concatenating files of different kinds — exactly as
football refuses to concatenate opening and closing odds.

*Non-finishers stay.* A fifth of a Grand Tour's start list can fail to reach
the end, and the abandons are not random — they concentrate among the riders
whose form was worst. Silently dropping them makes the remaining problem
easier than the real one and inflates every accuracy figure computed
afterwards. They are kept with `rank` NaN and a `status`, and `finishers()`
is the explicit opt-in.

*`time_seconds` is always a total, never a gap.* Results pages publish the
winner's elapsed time and everyone else's gap to it. Resolving that is the
scraper's job (see `cycling/scraper.py`); by the time a row reaches this
contract the column is total elapsed seconds or it is NaN. A frame where it
silently held gaps would look completely normal, so `time_order_violations`
looks for the fingerprint — a rider ranked behind another with a *smaller*
time — and the check warns about it rather than letting it pass unremarked.
"""

import os
import warnings

import numpy as np
import pandas as pd

from cycling.common import (
    FINISHED,
    NR,
    ONE_DAY,
    RESULT_COLUMNS,
    RESULT_KINDS,
    STATUSES,
)

REQUIRED_COLUMNS = ("Date", "Race", "Kind", "Rank", "Rider")
OPTIONAL_COLUMNS = ("Stage", "Team", "Status", "TimeSeconds")

# One result belongs to one (race, kind, stage). Every per-result invariant —
# unique ranks, times ordered by rank — is checked within these groups.
GROUP_KEYS = ["race", "kind", "stage"]

# Times are published to the second, and a resolved total is the sum of a
# whole-second winner time and a whole-second gap, so anything beyond this is
# a real ordering violation rather than rounding.
TIME_TOLERANCE_SECONDS = 1.0


class ResultFormatError(ValueError):
    """The file does not carry what a result frame needs, and guessing would be worse."""


def _parse_dates(raw):
    """dd/mm/yyyy, day-first, checked rather than trusted.

    A hand-edited file can carry a stray format, so pandas falling back to
    per-element parsing is an expected path here rather than something to fix —
    the result is validated below, which is what actually matters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    if parsed.isna().any():
        bad = pd.Series(raw)[parsed.isna()].head(3).tolist()
        raise ResultFormatError(
            f"{int(parsed.isna().sum())} date(s) could not be parsed as day-first, e.g. {bad}. "
            "The contract is dd/mm/yyyy, as written by cycling/scraper.py."
        )
    return parsed


def _group_index(results):
    """The (race, kind, stage) group each row belongs to, NaN stage included.

    `groupby` drops NaN keys by default and a one-day race has no stage, so the
    stage is filled with a sentinel here rather than losing every one-day row
    from every check that groups.
    """
    keys = zip(results["race"], results["kind"], results["stage"].fillna(-1))
    return pd.Series(list(keys), index=results.index, dtype=object)


def preprocess_results(df, validate=True):
    """Turn raw result rows into the tidy shape used everywhere else.

    Returns a frame with `RESULT_COLUMNS` and `results.attrs["result_kind"]`.
    Raises when the contract is broken — a mixed frame, a missing column, a
    rank that contradicts its status, two riders sharing a rank — and warns
    (through `check_result_format`) when the data is merely weak.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ResultFormatError(
            f"Missing required columns: {missing}. A result file carries "
            f"{list(REQUIRED_COLUMNS)}, optionally {list(OPTIONAL_COLUMNS)}."
        )

    out = pd.DataFrame({
        "ds": _parse_dates(df["Date"]),
        "race": df["Race"].astype("string").str.strip(),
        "kind": df["Kind"].astype("string").str.strip().str.lower(),
        "rider": df["Rider"].astype("string").str.strip(),
    })
    out["team"] = (df["Team"].astype("string").str.strip()
                   if "Team" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string"))
    out["stage"] = (pd.to_numeric(df["Stage"], errors="coerce")
                    if "Stage" in df.columns else np.nan)
    out["rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    out["time_seconds"] = (pd.to_numeric(df["TimeSeconds"], errors="coerce")
                           if "TimeSeconds" in df.columns else np.nan)

    if "Status" in df.columns:
        out["status"] = df["Status"].astype("string").str.strip().str.upper()
        # A blank status with a rank is a finisher; a blank status with no rank
        # says only that the rider is not in the classification.
        blank = out["status"].isna() | (out["status"] == "")
        out.loc[blank, "status"] = np.where(out.loc[blank, "rank"].notna(), FINISHED, NR)
    else:
        out["status"] = np.where(out["rank"].notna(), FINISHED, NR)
        if validate:
            warnings.warn(
                "No Status column: every ranked rider is taken as a finisher and every "
                "unranked one as 'not ranked'. That loses the reason for each abandon, "
                "which is the part a model of attrition would need.",
                stacklevel=2,
            )

    unknown_status = sorted(set(out.loc[~out["status"].isin(STATUSES), "status"].dropna()))
    if unknown_status:
        raise ResultFormatError(
            f"Unknown status value(s) {unknown_status}. Known: {list(STATUSES)}."
        )

    unknown_kind = sorted(set(out.loc[~out["kind"].isin(RESULT_KINDS), "kind"].dropna()))
    if unknown_kind:
        raise ResultFormatError(
            f"Unknown result kind(s) {unknown_kind}. Known: {list(RESULT_KINDS)}."
        )

    kinds = sorted(set(out["kind"].dropna()))
    if len(kinds) > 1:
        raise ResultFormatError(
            f"This frame mixes result kinds {kinds}. A rank in a stage result and a rank in "
            "a general classification are different quantities — one is a day's placing, the "
            "other is accumulated time — so one frame holds one kind. Split the file, or "
            "load the kinds separately."
        )
    kind = kinds[0] if kinds else None

    # A stage race's GC rows legitimately carry a stage number (the standing
    # after stage N); a one-day race has no stage at all, so a number there
    # means the rows are mislabelled.
    if kind == ONE_DAY and out["stage"].notna().any():
        raise ResultFormatError(
            "A one-day race has no stage number, but this frame carries one. Either the "
            "rows are stage results labelled one_day, or the stage column was filled in "
            "by hand; both change what `rank` means."
        )

    # A rank and a non-finishing status contradict each other, and there is no
    # safe way to pick a winner between them: dropping the rank invents an
    # abandon, keeping it invents a placing.
    contradictions = out["rank"].notna() & (out["status"] != FINISHED)
    if contradictions.any():
        rows = out.loc[contradictions, ["race", "rider", "rank", "status"]].head(3)
        raise ResultFormatError(
            f"{int(contradictions.sum())} row(s) carry both a rank and a non-finishing "
            f"status, e.g.\n{rows.to_string(index=False)}\nOne of the two is wrong."
        )
    unranked_finishers = out["rank"].isna() & (out["status"] == FINISHED)
    if unranked_finishers.any():
        raise ResultFormatError(
            f"{int(unranked_finishers.sum())} row(s) are marked as finishers with no rank. "
            "A finisher is by definition in the classification; an unranked rider needs the "
            "status that says why (DNF, DNS, DSQ, OTL or NR)."
        )

    out["group"] = _group_index(out)
    ranked = out[out["rank"].notna()]
    duplicated = ranked.duplicated(subset=["group", "rank"], keep=False)
    if duplicated.any():
        rows = ranked.loc[duplicated, ["race", "stage", "rank", "rider"]].head(4)
        raise ResultFormatError(
            f"{int(duplicated.sum())} row(s) share a rank within one result:\n"
            f"{rows.to_string(index=False)}\n"
            "Cycling results are ordered, so this is a parse error rather than a tie — check "
            "the scraped table before trusting the file."
        )

    out = (out.drop(columns="group")
              .sort_values(["ds", "race", "stage", "rank"], na_position="last")
              .reset_index(drop=True))
    out = out[RESULT_COLUMNS]
    out.attrs["result_kind"] = kind

    if validate:
        report = check_result_format(out)
        if report:
            warnings.warn(report["message"], stacklevel=2)
    return out


def time_order_violations(results):
    """Rows whose time is *faster* than a better-placed rider's in the same result.

    That cannot happen in a real classification, so it is the fingerprint of
    gaps stored in a column that is supposed to hold totals: a gap of 14
    seconds sitting under a winner's 4:15:22 puts rank 2 four hours ahead of
    rank 1. Returns a boolean Series aligned to `results`.
    """
    violation = pd.Series(False, index=results.index)
    if "time_seconds" not in results.columns:
        return violation

    timed = results[results["rank"].notna() & results["time_seconds"].notna()]
    for _, group in timed.groupby(_group_index(timed), sort=False):
        ordered = group.sort_values("rank")
        best_so_far = ordered["time_seconds"].cummax()
        behind = ordered["time_seconds"] < (best_so_far - TIME_TOLERANCE_SECONDS)
        violation.loc[ordered.index[behind]] = True
    return violation


def check_result_format(results):
    """Report on anything that weakens this frame. None when it is fit to use.

    Nothing here is fatal, which is the same line `lottery/utils/processor.py`
    draws: these rows are real results, and a caller may want them anyway.
    """
    problems = {}
    n = len(results)
    if n == 0:
        return {"n_results": 0, "message": "The frame is empty: no result rows at all."}

    violations = time_order_violations(results)
    if violations.any():
        problems["time_order"] = (
            f"{int(violations.sum())} rider(s) are timed faster than someone placed ahead of "
            "them. `time_seconds` is meant to be total elapsed time; this is what a column of "
            "gaps looks like. Do not use the times until the scrape is checked."
        )

    ranked = results["rank"].notna()
    missing_times = int((ranked & results["time_seconds"].isna()).sum())
    if missing_times:
        problems["times"] = (
            f"{missing_times} of {int(ranked.sum())} ranked rider(s) have no time. Gaps that "
            "could not be anchored to a winner's time are left NaN on purpose — a partial "
            "time column is visible, an invented one is not."
        )

    non_finishers = int((results["status"] != FINISHED).sum())
    if non_finishers == 0:
        problems["attrition"] = (
            "Not one non-finisher in the whole frame. A real race has abandons, so this file "
            "was probably filtered before it got here — which makes any accuracy measured on "
            "it optimistic, since the riders hardest to predict are the ones missing."
        )

    if results["team"].isna().all():
        problems["teams"] = "No team is recorded for any rider."

    if not problems:
        return None
    return {
        "result_kind": results.attrs.get("result_kind"),
        "n_results": n,
        "n_races": int(results["race"].nunique()),
        "n_non_finishers": non_finishers,
        "n_time_order_violations": int(violations.sum()),
        "message": " ".join(problems.values()),
    }


def load_and_preprocess(path, validate=True, finishers_only=False):
    """Read one result CSV into the tidy shape.

    `finishers_only=True` drops the non-finishers **after** the checks have run,
    so the warning about a frame with no abandons still fires on the file as
    published rather than on the filtered view.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    results = preprocess_results(pd.read_csv(path), validate=validate)
    if finishers_only:
        kind = results.attrs.get("result_kind")
        results = results[results["status"] == FINISHED].reset_index(drop=True)
        results.attrs["result_kind"] = kind
    return results


def load_races(paths, validate=True, finishers_only=False):
    """Concatenate several result files, refusing to merge different kinds.

    This is where the mixing would actually happen: a race's stage results and
    its general classification sit next to each other in the same directory,
    named alike, and stacking them produces a frame in which `rank` is a day's
    placing for some rows and a three-week standing for others.
    """
    frames = [load_and_preprocess(p, validate=validate, finishers_only=finishers_only)
              for p in paths]
    if not frames:
        raise ResultFormatError("No result files given.")

    kinds = {f.attrs.get("result_kind") for f in frames}
    if len(kinds) > 1:
        raise ResultFormatError(
            f"These files hold different result kinds ({sorted(str(k) for k in kinds)}), so "
            "concatenating them would put two different quantities in one `rank` column. "
            "Load one kind at a time."
        )

    merged = (pd.concat(frames, ignore_index=True)
                .sort_values(["ds", "race", "stage", "rank"], na_position="last")
                .reset_index(drop=True))
    merged.attrs.update(frames[0].attrs)
    return merged
