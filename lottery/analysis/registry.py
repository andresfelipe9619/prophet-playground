"""Pre-registration: a timestamped, append-only log of predictions made *before* the draw.

The reasoning, the three refusals and the storage now live in
[`core/registry.py`](../../core/registry.py); this module is the lottery half
of that contract. What it supplies is what `core/` deliberately lacks: what a
prediction *is* here (five main numbers plus a superbalota), how to score one
against a draw that has happened, and what to report beside the verdict.

The lift was not cosmetic. The lottery is the one domain where everybody
already knows the answer is no, and a forward record meant nothing here beyond
demonstrating the discipline. Football and cycling are where such a record
would be evidence, and they had none — so the machinery moved out and they
grew their own adapters beside this one.

The file lives at the repo root and is **not** gitignored, unlike
`exported_data/`. That is deliberate: committing it puts each prediction in
version control with a date attached, which is a stronger claim than any
timestamp column the file writes about itself.

At three draws a week, a year of this is 156 honest observations — enough, per
`lottery/analysis/power.py`, to detect an edge of about +23% and nothing subtler. Worth
knowing before you start, and it is why the summary reports the minimum
detectable effect alongside the result. `core/registry.py:status` deliberately
reports no such thing, because what counts as resolution is a domain question.
"""

import numpy as np
import pandas as pd

from core import registry as core_registry
from core.registry import RegistryError, RegistrySchema
from lottery.analysis.power import minimum_detectable_effect
from lottery.analysis.tickets import Ticket, check_ticket, draw_from_row
from lottery.models.baseline import beats_chance_test, expected_super_match_rate
from lottery.models.common import MAIN_BALLS_DRAWN

DEFAULT_REGISTRY_PATH = "predictions.csv"

# What a Baloto prediction is, and what scoring it produces. `core/registry.py`
# assembles the column order from this and never looks inside any of it.
SCHEMA = RegistrySchema(
    event_column="draw_date",       # must be in the future when recorded
    prediction_columns=("main", "super_ball"),   # main is "3-12-19-27-41"
    result_columns=("actual_main", "actual_super", "main_matches", "super_match"),
)

COLUMNS = list(SCHEMA.columns)

# Everything a pending row leaves blank, kept as a module constant because the
# dashboard and the tests both read it.
RESULT_COLUMNS = list(SCHEMA.late_columns)

__all__ = ["COLUMNS", "DEFAULT_REGISTRY_PATH", "RESULT_COLUMNS", "RegistryError",
           "load", "pending", "record", "record_predictions", "score_pending",
           "status", "summary"]


def load(path=DEFAULT_REGISTRY_PATH):
    """Read the registry, or an empty frame with the right columns if it does not exist yet."""
    return core_registry.load(SCHEMA, path)


def record(ticket, draw_date, label, note="", path=DEFAULT_REGISTRY_PATH, now=None):
    """Register one prediction for a future draw. Refuses anything that weakens the record.

    `now` is injectable so the guard itself can be tested; leave it alone in
    normal use, where letting the caller choose "now" would defeat the point.
    """
    if not isinstance(ticket, Ticket):
        raise TypeError(f"ticket must be a Ticket, got {type(ticket).__name__}")

    return core_registry.record(
        SCHEMA,
        {"main": "-".join(str(n) for n in sorted(ticket.main)),
         "super_ball": ticket.super_ball},
        draw_date, label, path=path, note=note, now=now,
    )


def record_predictions(predictions_by_position, draw_date, label, rng=None, **kwargs):
    """Convenience wrapper for a model's `{position: number}` output.

    Collisions between positions are filled at random by `ticket_from_predictions`,
    so the registered ticket may contain numbers the model did not choose. Say so
    in `note` when it matters — the registry records what was played, not what
    was predicted, and those differ exactly when the model had no signal.
    """
    from lottery.analysis.tickets import ticket_from_predictions
    return record(ticket_from_predictions(predictions_by_position, rng=rng),
                  draw_date, label, **kwargs)


def score_pending(df, balls_expanded, path=DEFAULT_REGISTRY_PATH):
    """Fill in results for every registered draw that has since happened.

    All of them, every time — there is no argument for scoring a subset, because
    picking which predictions to count is the failure this whole module exists
    to prevent. That rule lives in `core/registry.py`; what this function
    supplies is the lottery's `resolve`, which returns a result for a draw that
    has happened and None for one that has not.
    """
    by_date = {pd.Timestamp(date).normalize(): index for index, date in enumerate(df["ds"])}

    def resolve(row):
        position = by_date.get(row["draw_date"])
        if position is None:
            return None  # the draw has not happened, or is not in this history
        main_drawn, super_drawn = draw_from_row(balls_expanded.iloc[position])
        ticket = Ticket(main=tuple(int(n) for n in str(row["main"]).split("-")),
                        super_ball=int(row["super_ball"]))
        outcome = check_ticket(ticket, main_drawn, super_drawn)
        return {
            "actual_main": "-".join(str(n) for n in sorted(main_drawn)),
            "actual_super": int(super_drawn),
            "main_matches": outcome["main_matches"],
            "super_match": outcome["super_match"],
        }

    return core_registry.score_pending(SCHEMA, path, resolve)


def summary(path=DEFAULT_REGISTRY_PATH, registry=None, by_label=False):
    """Aggregate the scored predictions against the chance baseline.

    Reports the minimum detectable effect beside the p-value, because a young
    registry cannot say much and should say so: 20 predictions cannot detect an
    edge below about +65%, so "not beating chance" there is a statement about
    the sample size, not about the predictions.
    """
    registry = load(path) if registry is None else registry
    scored = registry[registry["main_matches"].notna()]
    if scored.empty:
        return pd.DataFrame(columns=["label", "n_scored", "avg_main_matches",
                                     "chance_avg_main_matches", "effect", "ci_low", "ci_high",
                                     "p_value_better_than_chance", "min_detectable_effect",
                                     "super_hit_rate", "chance_super_hit_rate"])

    groups = scored.groupby("label") if by_label else [("all", scored)]
    rows = []
    for label, group in groups:
        hits = group["main_matches"].astype(float).to_numpy()
        chance = beats_chance_test(hits, MAIN_BALLS_DRAWN)
        rows.append({
            "label": label,
            "n_scored": len(group),
            "avg_main_matches": float(hits.mean()),
            "chance_avg_main_matches": chance["chance_mean"],
            "effect": chance["effect"],
            "ci_low": chance["ci_low"],
            "ci_high": chance["ci_high"],
            "p_value_better_than_chance": chance["p_value_greater"],
            "min_detectable_effect": minimum_detectable_effect(len(group))["relative"],
            "super_hit_rate": float(group["super_match"].astype(str).str.lower()
                                    .isin(("true", "1")).mean()),
            "chance_super_hit_rate": expected_super_match_rate(),
        })
    return pd.DataFrame(rows).sort_values("avg_main_matches", ascending=False).reset_index(drop=True)


def pending(path=DEFAULT_REGISTRY_PATH, registry=None):
    """Predictions whose draw has not happened, or has not been scored yet."""
    return core_registry.pending(SCHEMA, path=path, registry=registry)


def status(path=DEFAULT_REGISTRY_PATH):
    """Counts plus what the registry could currently prove, for a one-glance answer.

    `core/registry.py:status` deliberately reports no resolution, because what
    counts as resolution is a domain question; the minimum detectable effect is
    this domain's answer and is added here.
    """
    state = core_registry.status(SCHEMA, path)
    n_scored = state["n_scored"]
    next_draw = state.pop("next_event")   # popped, not shadowed: one name per fact
    return {
        **state,
        "next_draw": next_draw,
        "min_detectable_effect": (minimum_detectable_effect(n_scored)["relative"]
                                  if n_scored else np.nan),
    }


if __name__ == "__main__":
    import argparse

    from lottery.models.common import (
        DEFAULT_DATA_PATH,
        infer_draw_weekdays,
        next_draw_dates,
    )
    from lottery.utils.processor import load_and_preprocess

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["record", "score", "show"])
    parser.add_argument("--file", default=DEFAULT_DATA_PATH)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--label", help="which model or strategy produced it")
    parser.add_argument("--main", help="5 numbers, e.g. 3-12-19-27-41")
    parser.add_argument("--super", dest="super_ball", type=int)
    parser.add_argument("--draw-date", help="YYYY-MM-DD; defaults to the next scheduled draw")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    pd.set_option("display.width", 150)

    if args.command == "record":
        if not (args.label and args.main and args.super_ball):
            raise SystemExit("record needs --label, --main and --super")
        df, _ = load_and_preprocess(args.file, validate=False, current_format_only=True)
        draw_date = args.draw_date or next_draw_dates(
            df["ds"].max(), 1, weekdays=infer_draw_weekdays(df["ds"]))[0]
        ticket = Ticket(main=tuple(int(n) for n in args.main.replace(",", "-").split("-")),
                        super_ball=args.super_ball)
        try:
            row = record(ticket, draw_date, args.label, note=args.note, path=args.registry)
        except RegistryError as exc:
            raise SystemExit(f"Refused: {exc}") from exc
        print(f"Recorded {row['main']} + {row['super_ball']} for {row['draw_date']:%Y-%m-%d} "
              f"as {row['label']!r} at {row['recorded_at']}.")

    elif args.command == "score":
        df, balls_expanded = load_and_preprocess(args.file, validate=False, current_format_only=True)
        registry = score_pending(df, balls_expanded, path=args.registry)
        print(f"{int(registry['main_matches'].notna().sum())} of {len(registry)} predictions scored.")

    state = status(args.registry)
    print(f"\n=== Registry: {state['n_recorded']} recorded, {state['n_scored']} scored, "
          f"{state['n_pending']} pending ===")
    table = summary(args.registry, by_label=True)
    if table.empty:
        print("Nothing scored yet — the registry proves nothing until its draws have happened.")
    else:
        print(table.to_string(index=False, formatters={
            "avg_main_matches": "{:.3f}".format, "chance_avg_main_matches": "{:.3f}".format,
            "effect": "{:+.3f}".format, "ci_low": "{:+.3f}".format, "ci_high": "{:+.3f}".format,
            "p_value_better_than_chance": "{:.3f}".format,
            "min_detectable_effect": "{:.0%}".format,
            "super_hit_rate": "{:.3f}".format, "chance_super_hit_rate": "{:.3f}".format,
        }))
        print("\nRead min_detectable_effect first: it is the smallest edge this many scored "
              "predictions could have revealed. Below it, 'no edge' only means 'too few draws yet'.")

    upcoming = pending(args.registry)
    if not upcoming.empty:
        print(f"\nPending ({len(upcoming)}):")
        print(upcoming[["draw_date", "label", "main", "super_ball", "recorded_at"]]
              .to_string(index=False))
