"""Football's model-vs-market backtest as a background job.

The domain half of `core/jobs.py` for football. The one job here is
`compare_models`, the walk-forward that refits Dixon-Coles once per held-out
match and is the slowest thing on the football page.

It is also the one that most needs to leave the request. A season of one league
can only resolve an edge of about 0.010 RPS (`football/power.py`), and the page
defaults to 40 matches; the only way the football verdict gets sharp enough to
mean anything is a run over many seasons, which is minutes, not seconds.

The match frame travels with the job, **attrs included**: `odds_source` and
`odds_are_closing` are what `market.py` and the page read to decide which bar
the model is being held to, and a worker that lost them would score against the
wrong market or none. `core/jobs.py` refuses any input that does not come back
out of the store exactly, attrs among it.
"""

from football.backtest import MODEL_NAMES, compare_models
from football.market import METHODS

COMPARE_MODELS = "football.compare_models"

# The training floor the page and the nightly run share: never fit on fewer
# than this many matches.
MIN_TRAIN = 100


def default_compare(method=METHODS[0]):
    """The market test the football page opens on.

    Shared with the nightly run for the reason `lottery/jobs.py` gives: one
    definition, so a result computed overnight is the one the page reads.
    """
    return {"n_windows": 40, "min_train": MIN_TRAIN, "half_life": 180, "method": method,
            "models": list(MODEL_NAMES), "blend_weight": 0.5, "pool": "linear"}


def full_history_windows(n_matches):
    """Every match after the training floor — the evaluation this module exists for.

    A season of one league resolves an edge of about 0.010 RPS at best; the
    page's 40-match default resolves far less. Scoring every held-out match is
    minutes of refitting, which is exactly what a background job is for.
    """
    return max(0, int(n_matches) - MIN_TRAIN)


def run_compare_models(params, inputs, progress):
    """`compare_models` with the page's parameters, one row per model."""
    models = tuple(params["models"])
    progress(f"Refitting {', '.join(models)} over {params['n_windows']} held-out matches")
    half_life = params.get("half_life")
    table = compare_models(inputs["matches"], n_windows=int(params["n_windows"]),
                           min_train=int(params.get("min_train", 100)),
                           half_life=int(half_life) if half_life else None,
                           method=params["method"], models=models,
                           blend_weight=float(params["blend_weight"]), pool=params["pool"])
    return {"table": table}


HANDLERS = {COMPARE_MODELS: run_compare_models}
