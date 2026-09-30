"""Cycling's walk-forward against the ranking as a background job.

The domain half of `core/jobs.py` for cycling: `compare_forecasters` with the
model, the ranking baseline and the uniform draw, exactly as the evaluation tab
builds it. The uniform draw stays in on purpose — it is the demonstration that
it is not a baseline — so the job has no option to leave it out.

Forecasters are named in the parameters rather than passed as functions,
because a job's parameters are JSON and a lambda is not. `FORECASTERS` is the
one place that turns a name back into the function, so the page and the worker
cannot disagree about what "plackett_luce" means.
"""

from cycling.baseline import form_worths, uniform_worths
from cycling.evaluation import compare_forecasters
from cycling.plackett_luce import PlackettLuce
from cycling.scoring import DEFAULT_METRIC

COMPARE_FORECASTERS = "cycling.compare_forecasters"


def default_min_history(n_races):
    """How many races of history the cycling evaluation opens on, for `n_races`."""
    return min(4, max(2, int(n_races) // 2))


def default_compare(n_races):
    """The walk-forward the cycling page opens on; shared with the nightly run."""
    return {"metric": DEFAULT_METRIC, "min_history": default_min_history(n_races)}

FORECASTERS = {
    "ranking": lambda history, riders, as_of: form_worths(history, riders, as_of=as_of),
    "plackett_luce": lambda history, riders, as_of: PlackettLuce.fit(history).worths_for(riders),
    "uniform": lambda history, riders, as_of: uniform_worths(len(riders)),
}


def run_compare_forecasters(params, inputs, progress):
    """The model and the uniform draw, each against the ranking, race by race."""
    progress(f"Refitting race by race after {params['min_history']} races of history")
    table, scores = compare_forecasters(inputs["results"], FORECASTERS, baseline="ranking",
                                        metric=params["metric"],
                                        min_history=int(params["min_history"]))
    return {"table": table, "scores": scores}


HANDLERS = {COMPARE_FORECASTERS: run_compare_forecasters}
