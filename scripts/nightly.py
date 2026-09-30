"""The scheduled run: check the data, score what has resolved, precompute the evaluations.

    python -m scripts.nightly                  # check, score, queue
    python -m scripts.nightly --run            # ...and run the queued jobs now, in this process

Meant for cron (or any scheduler) on the machine the dashboard runs on, e.g.

    15 3 * * *  cd /path/to/prophet-playground && venv/bin/python -m scripts.nightly --run

Three steps, each reported as OK, SKIP or FAIL, and the exit code is 1 if any
step FAILed — a schedule that only logs hides exactly the failures it exists to
catch:

1. **Check the draw store** (`scripts/store_sync.py check`): missing,
   mis-versioned, or no new draw in `--max-age-days`. A scraper whose markup
   changed keeps exiting 0 and appending nothing.
2. **Score every registry prediction that has resolved**, in all three domains,
   against that domain's bar — chance, the closing price, the pre-race ranking.
3. **Queue each domain's default evaluation** with the parameters and data the
   dashboard opens on (`default_*` in each domain's `jobs.py`), so the page finds
   the answer already stored instead of starting a run. Football also queues
   the full-history run, the one that can actually resolve an edge.

A step with nothing to act on — no store, no registry file, no data — is a SKIP
and says why, not a silent pass: "nothing happened" and "nothing needed to" are
different reports.
"""

import argparse
import importlib.util
import os
from dataclasses import dataclass

from core import jobs

OK, SKIP, FAIL = "OK", "SKIP", "FAIL"


@dataclass
class Step:
    name: str
    status: str
    detail: str


# ------------------------------------------------------------------ checks


def check_store(store, max_age_days):
    from scripts import store_sync

    if not os.path.exists(store):
        return Step("draw store", SKIP, f"no store at {store}; `python -m scripts.store_sync import` "
                                        "creates one")
    code = store_sync.main(["check", "--store", store, "--max-age-days", str(max_age_days)])
    return Step("draw store", OK if code == 0 else FAIL,
                "fresh and on the expected schema" if code == 0 else "see the FAIL line above")


# ------------------------------------------------------------------ registries


def _scored(status_before, status_after):
    return status_after["n_scored"] - status_before["n_scored"]


def score_lottery(registry_path, data_path):
    from lottery.analysis import registry
    from lottery.utils.processor import load_and_preprocess

    if not os.path.exists(registry_path):
        return Step("Baloto registry", SKIP, f"no registry at {registry_path}")
    if not os.path.exists(data_path):
        return Step("Baloto registry", SKIP, f"no draws at {data_path} to score against")
    before = registry.status(registry_path)
    df, balls_expanded = load_and_preprocess(data_path, validate=False, current_format_only=True)
    registry.score_pending(df, balls_expanded, path=registry_path)
    after = registry.status(registry_path)
    return Step("Baloto registry", OK,
                f"{_scored(before, after)} newly scored, {after['n_pending']} still pending")


def _season_paths(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(os.path.join(directory, name) for name in os.listdir(directory)
                  if name.lower().endswith(".csv"))


def score_football(registry_path, data_dir):
    from football import registry
    from football.market import market_probabilities
    from football.processor import load_seasons

    if not os.path.exists(registry_path):
        return Step("football registry", SKIP, f"no registry at {registry_path}")
    paths = _season_paths(data_dir)
    if not paths:
        return Step("football registry", SKIP, f"no season files in {data_dir}")
    before = registry.status(registry_path)
    # Closing-odds seasons only: a registered forecast is scored against the
    # closing price, and load_seasons refuses to mix sources anyway.
    matches = market_probabilities(load_seasons(paths, validate=False, closing_odds_only=True))
    registry.score_pending(matches, path=registry_path)
    after = registry.status(registry_path)
    return Step("football registry", OK,
                f"{_scored(before, after)} newly scored, {after['n_pending']} still pending")


def score_cycling(registry_path, data_dir):
    from cycling import registry
    from cycling.baseline import form_worths
    from cycling.processor import load_races

    if not os.path.exists(registry_path):
        return Step("cycling registry", SKIP, f"no registry at {registry_path}")
    paths = _season_paths(data_dir)
    if not paths:
        return Step("cycling registry", SKIP, f"no result files in {data_dir}")
    before = registry.status(registry_path)
    registry.score_pending(load_races(paths, validate=False), form_worths, path=registry_path)
    after = registry.status(registry_path)
    return Step("cycling registry", OK,
                f"{_scored(before, after)} newly scored, {after['n_pending']} still pending")


# ------------------------------------------------------------------ evaluations
#
# Each of these loads the data the way its dashboard page does by default and
# asks the question the page asks by default. If either drifts, nothing breaks —
# the page simply does not find the stored answer and offers to compute it —
# which is why the defaults live in one place, each domain's jobs.py.


def queue_lottery(store, data_path):
    import lottery.jobs as lottery_jobs
    from lottery.utils.processor import load_and_preprocess

    if not os.path.exists(data_path):
        return [Step("Baloto backtest", SKIP, f"no draws at {data_path}")]
    df, balls_expanded = load_and_preprocess(data_path, validate=False, current_format_only=True)
    params = lottery_jobs.default_walk_forward(
        len(df), include_prophet=importlib.util.find_spec("prophet") is not None)
    job = jobs.submit(store, lottery_jobs.WALK_FORWARD, params,
                      {"draws": lottery_jobs.draws_input(df, balls_expanded)},
                      label=f"Baloto · backtest de {params['n_windows']} ventanas (nocturno)")
    return [Step("Baloto backtest", OK, f"job {job['id']} ({job['status']})")]


def queue_football(store, data_dir):
    import football.jobs as football_jobs
    from football.processor import load_seasons

    paths = _season_paths(data_dir)
    if not paths:
        return [Step("football backtest", SKIP, f"no season files in {data_dir}")]
    # The page opens on the newest season file alone, with every source allowed.
    matches = load_seasons(paths[-1:], validate=False, closing_odds_only=False)
    if matches.attrs.get("odds_source") is None or len(matches) < 150:
        return [Step("football backtest", SKIP,
                     f"{os.path.basename(paths[-1])} has no odds or fewer than 150 matches")]
    steps = []
    default = football_jobs.default_compare()
    full = {**default, "n_windows": football_jobs.full_history_windows(len(matches))}
    for name, params in (("football backtest", default), ("football full history", full)):
        job = jobs.submit(store, football_jobs.COMPARE_MODELS, params, {"matches": matches},
                          label=f"Fútbol · {len(params['models'])} modelo(s) contra el mercado, "
                                f"{params['n_windows']} partidos (nocturno)")
        steps.append(Step(name, OK, f"job {job['id']} ({job['status']})"))
    return steps


def queue_cycling(store, data_dir):
    import cycling.jobs as cycling_jobs
    from cycling.evaluation import race_groups
    from cycling.processor import load_races

    paths = _season_paths(data_dir)
    if not paths:
        return [Step("cycling evaluation", SKIP, f"no result files in {data_dir}")]
    # The page opens on the first result file.
    results = load_races(paths[:1], validate=False)
    n_races = len(race_groups(results))
    if n_races < 5:
        return [Step("cycling evaluation", SKIP, f"only {n_races} scorable race(s) in {paths[0]}")]
    job = jobs.submit(store, cycling_jobs.COMPARE_FORECASTERS, cycling_jobs.default_compare(n_races),
                      {"results": results},
                      label=f"Ciclismo · modelo contra el ranking, {n_races} carreras (nocturno)")
    return [Step("cycling evaluation", OK, f"job {job['id']} ({job['status']})")]


# ------------------------------------------------------------------ the run


def _guarded(name, function, *args):
    """Run one step; an exception is that step's FAIL, never the whole night's."""
    try:
        result = function(*args)
    except Exception as exc:  # noqa: BLE001 — reported, and turned into the exit code
        return [Step(name, FAIL, f"{type(exc).__name__}: {exc}")]
    return result if isinstance(result, list) else [result]


def run(args, log=print):
    from cycling.common import DEFAULT_DATA_DIR as CYCLING_DIR
    from football.common import DEFAULT_DATA_DIR as FOOTBALL_DIR
    from lottery.models.common import DEFAULT_DATA_PATH

    lottery_data = args.lottery_data or DEFAULT_DATA_PATH
    football_dir = args.football_dir or FOOTBALL_DIR
    cycling_dir = args.cycling_dir or CYCLING_DIR

    steps = []
    steps += _guarded("draw store", check_store, args.store, args.max_age_days)
    steps += _guarded("Baloto registry", score_lottery, args.lottery_registry, lottery_data)
    steps += _guarded("football registry", score_football, args.football_registry, football_dir)
    steps += _guarded("cycling registry", score_cycling, args.cycling_registry, cycling_dir)
    steps += _guarded("Baloto backtest", queue_lottery, args.jobs, lottery_data)
    steps += _guarded("football backtest", queue_football, args.jobs, football_dir)
    steps += _guarded("cycling evaluation", queue_cycling, args.jobs, cycling_dir)

    if args.run:
        from scripts.worker import available_handlers

        ran = jobs.run_worker(args.jobs, available_handlers(log=log), once=True, log=log)
        failed = [j for j in jobs.list_jobs(args.jobs, limit=max(ran, 1)) if j["status"] == jobs.FAILED]
        steps.append(Step("run queued jobs", FAIL if failed else OK,
                          f"{ran} ran, {len(failed)} failed"
                          + (f" (#{', #'.join(str(j['id']) for j in failed)})" if failed else "")))

    for step in steps:
        log(f"{step.status:4}  {step.name}: {step.detail}")
    return 1 if any(step.status == FAIL for step in steps) else 0


def build_parser():
    from scripts.store_sync import DEFAULT_MAX_AGE_DAYS, DEFAULT_STORE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="run the queued jobs in this process before exiting")
    parser.add_argument("--jobs", default=jobs.DEFAULT_PATH, help="the job store")
    parser.add_argument("--store", default=DEFAULT_STORE, help="the draw store to check")
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--lottery-data", default=None,
                        help="draw CSV or store the Baloto page reads (default: its default)")
    parser.add_argument("--football-dir", default=None)
    parser.add_argument("--cycling-dir", default=None)
    parser.add_argument("--lottery-registry", default="predictions.csv")
    parser.add_argument("--football-registry", default="football_predictions.csv")
    parser.add_argument("--cycling-registry", default="cycling_predictions.csv")
    return parser


def main(argv=None, log=print):
    return run(build_parser().parse_args(argv), log=log)


if __name__ == "__main__":
    raise SystemExit(main())
