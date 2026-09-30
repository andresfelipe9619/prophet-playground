"""The scheduled run and the worker CLI — scripts/nightly.py and scripts/worker.py.

The property that makes a nightly run useful rather than busywork: **the job it
queues is the job the dashboard asks for**, same parameters and same data, so the
page reads the stored answer in the morning instead of starting a run. The page
itself cannot be imported here (streamlit is not a test dependency), so these
tests pin the half that can drift silently — that nightly loads each domain's
default data and builds each domain's default parameters through the same
functions the page calls.

And the reporting rule: a step with nothing to act on is a SKIP that says why,
a step that breaks is a FAIL that sets the exit code, and one broken step does
not stop the others from running.
"""

import pandas as pd
import pytest

import cycling.jobs as cycling_jobs
import football.jobs as football_jobs
import lottery.jobs as lottery_jobs
from core import jobs
from cycling.evaluation import race_groups
from cycling.processor import load_races
from cycling.sample_data import generate_stage_race
from football.processor import load_seasons
from football.sample_data import generate_matches
from lottery.utils.processor import load_and_preprocess
from lottery.utils.sample_data import generate_sample_draws
from scripts import nightly, worker


@pytest.fixture
def world(tmp_path):
    """One draw CSV, one football season and one cycling file, as they sit on disk."""
    lottery_csv = tmp_path / "draws.csv"
    generate_sample_draws(n_draws=120).to_csv(lottery_csv, index=False)

    football_dir = tmp_path / "football"
    football_dir.mkdir()
    # 14 teams play 182 matches: over the 150 the page needs for a backtest.
    generate_matches(n_teams=14, seed=3).drop(columns=["TrueH", "TrueD", "TrueA"]).to_csv(
        football_dir / "E0_2324.csv", index=False)

    cycling_dir = tmp_path / "cycling"
    cycling_dir.mkdir()
    generate_stage_race(seed=0, n_riders=20, n_stages=8).to_csv(
        cycling_dir / "tour_2024_stage.csv", index=False)

    return {"tmp": tmp_path, "lottery": str(lottery_csv), "football": str(football_dir),
            "cycling": str(cycling_dir), "jobs": str(tmp_path / "jobs.sqlite")}


def args(world, *extra):
    return ["--jobs", world["jobs"], "--store", str(world["tmp"] / "no-store.sqlite"),
            "--lottery-data", world["lottery"], "--football-dir", world["football"],
            "--cycling-dir", world["cycling"],
            "--lottery-registry", str(world["tmp"] / "predictions.csv"),
            "--football-registry", str(world["tmp"] / "football_predictions.csv"),
            "--cycling-registry", str(world["tmp"] / "cycling_predictions.csv"), *extra]


def run_quietly(argv):
    lines = []
    code = nightly.main(argv, log=lines.append)
    return code, lines


def test_every_step_reports_and_missing_inputs_are_skips_not_passes(world):
    code, lines = run_quietly(args(world))

    report = "\n".join(lines)
    assert code == 0
    assert "SKIP  draw store: no store at" in report
    assert "SKIP  Baloto registry: no registry at" in report
    assert "OK    Baloto backtest: job" in report
    assert "OK    football backtest: job" in report
    assert "OK    football full history: job" in report
    assert "OK    cycling evaluation: job" in report


def test_the_queued_baloto_job_is_the_one_the_page_asks_for(world):
    run_quietly(args(world))

    df, balls_expanded = load_and_preprocess(world["lottery"], validate=False,
                                             current_format_only=True)
    import importlib.util
    params = lottery_jobs.default_walk_forward(
        len(df), include_prophet=importlib.util.find_spec("prophet") is not None)
    key = jobs.job_key(lottery_jobs.WALK_FORWARD, params,
                       {"draws": lottery_jobs.draws_input(df, balls_expanded)})

    assert jobs.find(world["jobs"], key) is not None


def test_the_queued_football_jobs_are_the_default_and_the_full_history(world):
    run_quietly(args(world))

    matches = load_seasons([f"{world['football']}/E0_2324.csv"], validate=False)
    default = football_jobs.default_compare()
    full = {**default, "n_windows": football_jobs.full_history_windows(len(matches))}

    for params in (default, full):
        key = jobs.job_key(football_jobs.COMPARE_MODELS, params, {"matches": matches})
        assert jobs.find(world["jobs"], key) is not None
    assert full["n_windows"] == len(matches) - football_jobs.MIN_TRAIN


def test_the_queued_cycling_job_is_the_one_the_page_asks_for(world):
    run_quietly(args(world))

    results = load_races([f"{world['cycling']}/tour_2024_stage.csv"], validate=False)
    params = cycling_jobs.default_compare(len(race_groups(results)))
    key = jobs.job_key(cycling_jobs.COMPARE_FORECASTERS, params, {"results": results})

    assert jobs.find(world["jobs"], key) is not None


def test_running_nightly_twice_queues_nothing_new(world):
    """The second night on unchanged data finds its questions already asked."""
    run_quietly(args(world))
    before = jobs.counts(world["jobs"])
    run_quietly(args(world))

    assert jobs.counts(world["jobs"]) == before


def test_a_broken_step_fails_the_run_without_stopping_the_others(world):
    with open(f"{world['football']}/E0_2324.csv", "w", encoding="utf-8") as handle:
        handle.write("<html>503 Service Unavailable</html>\n")

    code, lines = run_quietly(args(world))

    report = "\n".join(lines)
    assert code == 1
    assert "FAIL  football backtest:" in report
    assert "OK    cycling evaluation: job" in report   # ran after the failure


def test_a_stale_draw_store_fails_the_night(world, tmp_path):
    from lottery.utils.processor import import_csv

    store = str(tmp_path / "draws.sqlite")
    import_csv(world["lottery"], store)   # sample draws end in 2018: very stale

    argv = args(world)
    argv[argv.index("--store") + 1] = store
    code, lines = run_quietly(argv)

    assert code == 1
    assert any(line.startswith("FAIL  draw store") for line in lines)


@pytest.mark.slow
def test_run_drains_the_queue_in_the_same_invocation(world):
    code, lines = run_quietly(args(world, "--run"))

    assert code == 0, "\n".join(lines)
    counts = jobs.counts(world["jobs"])
    assert counts[jobs.DONE] == 4 and counts[jobs.QUEUED] == 0
    assert any(line.startswith("OK    run queued jobs: 4 ran, 0 failed") for line in lines)


def test_registries_with_resolved_predictions_are_scored(world, tmp_path):
    from lottery.analysis import registry
    from lottery.analysis.tickets import Ticket

    path = str(tmp_path / "predictions.csv")
    registry.record(Ticket(main=(3, 12, 19, 27, 41), super_ball=8), "2031-01-01", "yo", path=path)
    frame = pd.read_csv(path)
    frame["draw_date"] = "2018-01-03"   # backdate to a draw that exists in the sample
    frame.to_csv(path, index=False)

    code, lines = run_quietly(args(world))

    assert any("Baloto registry: 1 newly scored, 0 still pending" in line for line in lines)
    assert code == 0


# ------------------------------------------------------------------ worker CLI

def test_the_worker_imports_every_domain_it_can(world):
    handlers = worker.available_handlers(log=lambda _: None)

    assert set(handlers) == {*lottery_jobs.HANDLERS, *football_jobs.HANDLERS, *cycling_jobs.HANDLERS}


def test_a_domain_that_cannot_be_imported_is_skipped_and_named():
    messages = []

    handlers = worker.available_handlers(("cycling.jobs", "no_such_domain.jobs"), log=messages.append)

    assert set(handlers) == set(cycling_jobs.HANDLERS)
    assert any("skipping no_such_domain.jobs" in m for m in messages)


def test_the_worker_refuses_a_kind_it_does_not_have(world):
    messages = []

    code = worker.main(["--store", world["jobs"], "--once", "--kinds", "tennis.elo"],
                       log=messages.append)

    assert code == 2
    assert any("tennis.elo" in m for m in messages)


def test_the_worker_once_runs_what_is_queued_and_exits(world):
    run_quietly(args(world))
    messages = []

    code = worker.main(["--store", world["jobs"], "--once", "--kinds",
                        cycling_jobs.COMPARE_FORECASTERS], log=messages.append)

    assert code == 0
    assert any(m == "ran 1 job(s)" for m in messages)
    done = jobs.list_jobs(world["jobs"], status=jobs.DONE)
    assert [j["kind"] for j in done] == [cycling_jobs.COMPARE_FORECASTERS]
