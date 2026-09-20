"""Predictions written down before the thing they predict happened.

Everything else in this project is retrospective, and retrospective analysis
can always be tuned after the fact — a window shifted, a model swapped, a run
quietly not counted. None of that is dishonesty; it is what analysing data you
have already seen does to anyone. The one thing that cannot be tuned afterwards
is a prediction recorded before the result existed.

This module is the domain-free half of that idea. It was lifted out of
`lottery/analysis/registry.py`, which had it first and which now sits on top of
it — and the lift matters because the lottery is the **one domain where
everybody already knows the answer is no**. Football and cycling are where a
forward record would mean something, and they had none.

**Three refusals, and none of them may soften into a warning.**

1. **A prediction for an event that has already happened is rejected.** Not
   warned about — rejected. A registry that accepts backdated entries proves
   nothing, and one such row makes the whole file worthless.
2. **Rows are append-only and never edited.** A second prediction for the same
   (event, label) is refused. Change your mind by registering under a different
   label, which leaves both on the record.
3. **Every eligible row is scored, or none are.** There is no way to score a
   subset, because choosing which predictions to count is exactly the failure
   mode this exists to prevent.

What the domain supplies is everything else: what an event is called, which
columns carry a prediction, which carry a result, and how to turn one into the
other. This module never inspects any of it — it only guarantees that what it
holds was unfalsifiable when it was written.

The file is CSV, sorted, and deliberately **not** gitignored by any domain
using it: committing it dates each prediction in version control, which is a
stronger claim than any timestamp a file writes about itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

# Filled by `score_pending` for every row, on top of the domain's own result
# columns, and doubling as the "has this been scored" flag. Every registry also
# carries `recorded_at`, `label` and `note`; those are assembled by
# `RegistrySchema.columns` rather than named here, because the order they sit
# in relative to the domain's own columns is part of the file's shape.
SCORED_AT = "scored_at"


class RegistryError(RuntimeError):
    """A write that would make the registry stop being evidence."""


@dataclass(frozen=True)
class RegistrySchema:
    """What a domain's registry holds, beyond the columns every registry has.

    `event_column` is what the domain calls the thing being predicted — a draw
    date, a kick-off, a race day. The file carries that name, not a generic one:
    a column headed `event_date` in a committed predictions file would be one
    more thing a reader has to translate, and this module never needs to know
    what the name means.

    `prediction_columns` are written at record time and never touched again.
    `result_columns` are blank until scoring fills them.
    """

    event_column: str
    prediction_columns: tuple[str, ...]
    result_columns: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column, in the order the file carries them."""
        return (
            "recorded_at", self.event_column, "label",
            *self.prediction_columns, "note",
            SCORED_AT, *self.result_columns,
        )

    @property
    def late_columns(self) -> tuple[str, ...]:
        """The columns a pending row leaves blank.

        Held as `object` rather than letting pandas infer: an unscored registry
        has these all-NA, pandas would type them float64, and the first real
        score would then be an incompatible-dtype assignment — a FutureWarning
        today and an error later.
        """
        return (SCORED_AT, *self.result_columns)


def _now() -> datetime:
    return datetime.now(UTC)


def _typed(schema: RegistrySchema, registry: pd.DataFrame) -> pd.DataFrame:
    registry = registry.copy()
    registry[schema.event_column] = pd.to_datetime(registry[schema.event_column])
    for column in schema.late_columns:
        registry[column] = registry[column].astype(object)
    return registry


def load(schema: RegistrySchema, path: str) -> pd.DataFrame:
    """Read the registry, or an empty frame with the right columns if it does not exist."""
    if not os.path.exists(path):
        empty = pd.DataFrame({column: pd.Series(dtype=object) for column in schema.columns})
        return _typed(schema, empty)

    registry = pd.read_csv(path)
    for column in schema.columns:
        if column not in registry.columns:
            registry[column] = pd.NA
    return _typed(schema, registry[list(schema.columns)])


def write(schema: RegistrySchema, registry: pd.DataFrame, path: str) -> None:
    """Sorted by event then label, so a diff of the committed file reads chronologically."""
    registry.sort_values([schema.event_column, "label"]).reset_index(drop=True).to_csv(
        path, index=False)


def record(schema: RegistrySchema, prediction: dict[str, Any], event_date: Any, label: str,
           path: str, note: str = "", now: datetime | None = None) -> dict[str, Any]:
    """Register one prediction for a future event, or refuse.

    `prediction` must carry exactly the schema's `prediction_columns` — a
    missing one would write a row that cannot be scored, and an extra one is a
    caller who thinks the schema says something it does not.

    `now` is injectable so the first refusal can be tested; leave it alone in
    normal use, where letting the caller choose "now" would defeat the point.
    """
    missing = [c for c in schema.prediction_columns if c not in prediction]
    extra = [c for c in prediction if c not in schema.prediction_columns]
    if missing or extra:
        raise RegistryError(
            f"A prediction must carry exactly {list(schema.prediction_columns)}; "
            f"missing {missing}, unexpected {extra}."
        )

    event_date = pd.Timestamp(event_date).normalize()
    moment = now or _now()
    today = pd.Timestamp(moment.date())
    if event_date <= today:
        raise RegistryError(
            f"{schema.event_column} {event_date:%Y-%m-%d} is not in the future "
            f"(today is {today:%Y-%m-%d}). A prediction recorded after its event proves "
            "nothing, so the registry will not hold one — every row in the file has to have "
            "been unfalsifiable when it was written."
        )

    registry = load(schema, path)
    clash = registry[(registry[schema.event_column] == event_date)
                     & (registry["label"] == label)]
    if not clash.empty:
        already = ", ".join(f"{c}={clash.iloc[0][c]}" for c in schema.prediction_columns)
        raise RegistryError(
            f"A prediction labelled {label!r} for {event_date:%Y-%m-%d} is already on the "
            f"record ({already}). Rows are append-only — register a revision under a "
            "different label so both stay visible."
        )

    row: dict[str, Any] = {
        "recorded_at": moment.isoformat(timespec="seconds"),
        schema.event_column: event_date,
        "label": label,
        **{column: prediction[column] for column in schema.prediction_columns},
        "note": note,
        SCORED_AT: pd.NA,
        **{column: pd.NA for column in schema.result_columns},
    }
    new_row = _typed(schema, pd.DataFrame([row]))
    updated = new_row if registry.empty else pd.concat([registry, new_row], ignore_index=True)
    write(schema, updated, path)
    return row


def score_pending(schema: RegistrySchema, path: str,
                  resolve: Callable[[Any], dict[str, Any] | None]) -> pd.DataFrame:
    """Fill in results for every registered event that has since happened.

    All of them, every time. `resolve(row)` returns a dict of the schema's
    `result_columns` for a row whose event has happened and whose result is
    known, or **None** when it has not — those two cases are the only ones, and
    a domain that wants to skip a row for any other reason is asking for the
    subset scoring this module exists to prevent.

    Already-scored rows are left untouched, so re-running is safe and cannot
    rewrite history.
    """
    registry = load(schema, path)
    if registry.empty:
        return registry

    scored = 0
    for i, row in registry.iterrows():
        if pd.notna(row[SCORED_AT]):
            continue
        result = resolve(row)
        if result is None:
            continue

        missing = [c for c in schema.result_columns if c not in result]
        if missing:
            raise RegistryError(
                f"Scoring returned no value for {missing}. A half-scored row reads as a "
                "scored one and would be counted as evidence."
            )

        registry.loc[i, SCORED_AT] = _now().isoformat(timespec="seconds")
        for column in schema.result_columns:
            registry.loc[i, column] = result[column]
        scored += 1

    if scored:
        write(schema, registry, path)
    return registry


def pending(schema: RegistrySchema, path: str | None = None,
            registry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predictions whose event has not happened, or has not been scored yet.

    Either a `path` to read or a `registry` already in hand — passing neither
    is a caller who has lost track of which, so it raises rather than quietly
    reading a default file.
    """
    if registry is None:
        if path is None:
            raise RegistryError("pending() needs either a path or a registry frame.")
        registry = load(schema, path)
    return registry[registry[SCORED_AT].isna()].sort_values(schema.event_column)


def status(schema: RegistrySchema, path: str) -> dict[str, Any]:
    """Counts and dates, for a one-glance answer.

    Deliberately says nothing about how good the predictions are. A domain adds
    that — the lottery attaches its minimum detectable effect, because a young
    registry cannot say much and should say so rather than let a null result
    read as a finding.
    """
    registry = load(schema, path)
    n_scored = int(registry[SCORED_AT].notna().sum())
    return {
        "path": path,
        "n_recorded": len(registry),
        "n_scored": n_scored,
        "n_pending": len(registry) - n_scored,
        "first_recorded": registry["recorded_at"].min() if len(registry) else None,
        "next_event": (pending(schema, registry=registry)[schema.event_column].min()
                       if len(registry) else None),
    }
