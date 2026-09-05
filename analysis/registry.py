"""Pre-registration: a timestamped, append-only log of predictions made *before* the draw.

Everything else in this project is retrospective, and retrospective analysis can
always be tuned after the fact — a window shifted, a model swapped, a run
quietly not counted. None of that is dishonesty; it is what analysing data you
have already seen does to anyone. The one thing that cannot be tuned afterwards
is a prediction written down before the result existed.

That is all this module is. Record a prediction against a future draw date, and
score it once the draw has happened. What makes it evidence rather than
bookkeeping is what it refuses to do:

- **A prediction for a draw that already happened is rejected.** Not warned
  about — rejected. A registry that accepts backdated entries proves nothing,
  and one entry is enough to make the whole file worthless.
- **Rows are append-only and never edited.** `record` refuses to write a second
  prediction for the same (draw date, label) pair. Change your mind by
  registering under a different label, which leaves both on the record.
- **Every row is scored, or none are.** `score_pending` fills in results for
  every draw that has since happened. There is no way to score selected rows,
  because choosing which predictions to count is exactly the failure mode.

The file lives at the repo root and is **not** gitignored, unlike
`exported_data/`. That is deliberate: committing it puts each prediction in
version control with a date attached, which is a stronger claim than any
timestamp column the file writes about itself.

At three draws a week, a year of this is 156 honest observations — enough, per
`analysis/power.py`, to detect an edge of about +23% and nothing subtler. Worth
knowing before you start, and it is why the summary reports the minimum
detectable effect alongside the result.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analysis.power import minimum_detectable_effect
from analysis.tickets import Ticket, check_ticket, draw_from_row
from models.baseline import beats_chance_test, expected_super_match_rate
from models.common import MAIN_BALLS_DRAWN

DEFAULT_REGISTRY_PATH = "predictions.csv"

COLUMNS = [
    "recorded_at",     # UTC timestamp, written by this module
    "draw_date",       # the draw being predicted — must be in the future when recorded
    "label",           # free text: which model, strategy or hunch produced it
    "main",            # "3-12-19-27-41"
    "super_ball",
    "note",
    "scored_at",       # filled by score_pending
    "actual_main",
    "actual_super",
    "main_matches",
    "super_match",
]

# Everything a pending row leaves blank. Held as `object` rather than letting
# pandas infer: an unscored registry has these all-NA, pandas would type them
# float64, and the first real score would then be an incompatible-dtype
# assignment — a FutureWarning today and an error later.
RESULT_COLUMNS = ["scored_at", "actual_main", "actual_super", "main_matches", "super_match"]


class RegistryError(RuntimeError):
    """A write that would make the registry stop being evidence."""


def _now():
    return datetime.now(timezone.utc)


def load(path=DEFAULT_REGISTRY_PATH):
    """Read the registry, or an empty frame with the right columns if it does not exist yet."""
    if not os.path.exists(path):
        return _typed(pd.DataFrame({column: pd.Series(dtype=object) for column in COLUMNS}))

    registry = pd.read_csv(path)
    for column in COLUMNS:
        if column not in registry.columns:
            registry[column] = pd.NA
    return _typed(registry[COLUMNS])


def _typed(registry):
    """Pin the columns whose values arrive late to `object`, and draw_date to a timestamp."""
    registry = registry.copy()
    registry["draw_date"] = pd.to_datetime(registry["draw_date"])
    for column in RESULT_COLUMNS:
        registry[column] = registry[column].astype(object)
    return registry


def record(ticket, draw_date, label, note="", path=DEFAULT_REGISTRY_PATH, now=None):
    """Register one prediction for a future draw. Refuses anything that weakens the record.

    `now` is injectable so the guard itself can be tested; leave it alone in
    normal use, where letting the caller choose "now" would defeat the point.
    """
    if not isinstance(ticket, Ticket):
        raise TypeError(f"ticket must be a Ticket, got {type(ticket).__name__}")

    draw_date = pd.Timestamp(draw_date).normalize()
    moment = now or _now()
    today = pd.Timestamp(moment.date())
    if draw_date <= today:
        raise RegistryError(
            f"Draw date {draw_date:%Y-%m-%d} is not in the future (today is {today:%Y-%m-%d}). "
            "A prediction recorded after its draw proves nothing, so the registry will not hold "
            "one — every row in the file has to have been unfalsifiable when it was written."
        )

    registry = load(path)
    clash = registry[(registry["draw_date"] == draw_date) & (registry["label"] == label)]
    if not clash.empty:
        raise RegistryError(
            f"A prediction labelled {label!r} for {draw_date:%Y-%m-%d} is already on the record "
            f"({clash.iloc[0]['main']} + {clash.iloc[0]['super_ball']}). Rows are append-only — "
            "register a revision under a different label so both stay visible."
        )

    row = {
        "recorded_at": moment.isoformat(timespec="seconds"),
        "draw_date": draw_date,
        "label": label,
        "main": "-".join(str(n) for n in sorted(ticket.main)),
        "super_ball": ticket.super_ball,
        "note": note,
        "scored_at": pd.NA, "actual_main": pd.NA, "actual_super": pd.NA,
        "main_matches": pd.NA, "super_match": pd.NA,
    }
    new_row = _typed(pd.DataFrame([row]))
    updated = new_row if registry.empty else pd.concat([registry, new_row], ignore_index=True)
    _write(updated, path)
    return row


def record_predictions(predictions_by_position, draw_date, label, rng=None, **kwargs):
    """Convenience wrapper for a model's `{position: number}` output.

    Collisions between positions are filled at random by `ticket_from_predictions`,
    so the registered ticket may contain numbers the model did not choose. Say so
    in `note` when it matters — the registry records what was played, not what
    was predicted, and those differ exactly when the model had no signal.
    """
    from analysis.tickets import ticket_from_predictions
    return record(ticket_from_predictions(predictions_by_position, rng=rng),
                  draw_date, label, **kwargs)


def _write(registry, path):
    registry = registry.sort_values(["draw_date", "label"]).reset_index(drop=True)
    registry.to_csv(path, index=False)


def score_pending(df, balls_expanded, path=DEFAULT_REGISTRY_PATH):
    """Fill in results for every registered draw that has since happened.

    All of them, every time — there is no argument for scoring a subset, because
    picking which predictions to count is the failure this whole module exists
    to prevent. Already-scored rows are left untouched, so re-running is safe
    and cannot rewrite history.
    """
    registry = load(path)
    if registry.empty:
        return registry

    results = {pd.Timestamp(date).normalize(): index
               for index, date in enumerate(df["ds"])}

    scored = 0
    for i, row in registry.iterrows():
        if pd.notna(row["scored_at"]) or row["draw_date"] not in results:
            continue

        main_drawn, super_drawn = draw_from_row(balls_expanded.iloc[results[row["draw_date"]]])
        ticket = Ticket(main=tuple(int(n) for n in str(row["main"]).split("-")),
                        super_ball=int(row["super_ball"]))
        outcome = check_ticket(ticket, main_drawn, super_drawn)

        registry.loc[i, "scored_at"] = _now().isoformat(timespec="seconds")
        registry.loc[i, "actual_main"] = "-".join(str(n) for n in sorted(main_drawn))
        registry.loc[i, "actual_super"] = int(super_drawn)
        registry.loc[i, "main_matches"] = outcome["main_matches"]
        registry.loc[i, "super_match"] = outcome["super_match"]
        scored += 1

    if scored:
        _write(registry, path)
    return registry


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
    registry = load(path) if registry is None else registry
    return registry[registry["main_matches"].isna()].sort_values("draw_date")


def status(path=DEFAULT_REGISTRY_PATH):
    """Counts plus what the registry could currently prove, for a one-glance answer."""
    registry = load(path)
    n_scored = int(registry["main_matches"].notna().sum())
    return {
        "path": path,
        "n_recorded": len(registry),
        "n_scored": n_scored,
        "n_pending": len(registry) - n_scored,
        "first_recorded": registry["recorded_at"].min() if len(registry) else None,
        "next_draw": pending(registry=registry)["draw_date"].min() if len(registry) else None,
        "min_detectable_effect": (minimum_detectable_effect(n_scored)["relative"]
                                  if n_scored else np.nan),
    }


if __name__ == "__main__":
    import argparse

    from models.common import DEFAULT_DATA_PATH, build_position_series, infer_draw_weekdays, next_draw_dates
    from utils.processor import load_and_preprocess

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
            raise SystemExit(f"Refused: {exc}")
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
