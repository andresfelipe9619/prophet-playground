"""The background job queue — core/jobs.py.

What these pin is the ways a queue goes quietly wrong, not the happy path:

- **Identity is what the job computes.** The same kind, parameters and data is
  one job however often it is submitted; one corrected cell is another job.
- **A result that would not come back as it went in is refused.** An object
  column of `None` read back from JSON is `float64`, and a store that accepted
  that would hand the next reader a different frame under the same name.
- **A dead worker does not leave a job running forever**, and a job that keeps
  killing its worker stops being retried.
- **A busy worker is not mistaken for a dead one.** The heartbeat runs from a
  thread, so a long fit with nothing to report still keeps its claim.
- **Only the owner can finish a job.** A worker whose job was requeued behind
  its back must not overwrite the second owner's answer.
"""

import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from core import jobs


@pytest.fixture
def store(tmp_path):
    return str(tmp_path / "jobs.sqlite")


def frame(values=(1, 2, 3)):
    return pd.DataFrame({"x": list(values), "label": [f"r{v}" for v in values]})


def echo(params, inputs, progress):
    """A handler that proves it saw its parameters and its stored input."""
    data = inputs["data"]
    progress("summing")
    return {"total": int(data["x"].sum()) * params.get("scale", 1),
            "table": data.assign(doubled=data["x"] * 2)}


def boom(params, inputs, progress):
    raise ValueError("the model did not converge")


# ------------------------------------------------------------ result encoding

def test_an_object_column_of_none_comes_back_as_object_not_float():
    """The dtype change found on the football backtest table: `half_life`,
    `pool` and `calibrate` hold None, and JSON reads that back as float64."""
    table = pd.DataFrame({"model": ["elo", "dixon_coles"], "half_life": [None, None],
                          "score": [0.2, 0.19]})

    back = jobs.decode_result(jobs.checked_encode({"table": table}))["table"]

    assert back["half_life"].dtype == object
    assert back["half_life"].tolist() == [None, None]
    pd.testing.assert_frame_equal(back, table)


def test_dates_numbers_and_attrs_survive_the_store():
    table = pd.DataFrame({"ds": pd.to_datetime(["2024-01-03", "2024-01-06"]),
                          "hits": [1, 2], "flag": [True, False]})
    table.attrs["manifest"] = {"generated_at": "now", "inputs": {"n": 2}}
    info = {"cutoff": pd.Timestamp("2024-01-01"), "n_train": 87, "missing": float("nan")}

    back = jobs.decode_result(jobs.checked_encode({"table": table, "info": info}))

    pd.testing.assert_frame_equal(back["table"], table)
    assert back["table"].attrs == table.attrs
    assert back["info"]["cutoff"] == pd.Timestamp("2024-01-01")
    assert back["info"]["n_train"] == 87
    assert pd.isna(back["info"]["missing"])


def test_floats_come_back_bit_identical_not_rounded():
    """`DataFrame.to_json` rounds to ten significant digits, and the first
    version of this store used it: the football worker fitted on odds that
    differed from the page's in the eleventh digit, and a tolerance-based guard
    called that equal. Exact is the only standard a stored input can meet."""
    values = [0.1 + 0.2, 1 / 3, 2.718281828459045, 1e-300, float("nan"), float("inf")]
    table = pd.DataFrame({"odds": values})

    back = jobs.decode_result(jobs.checked_encode({"table": table}))["table"]

    assert back["odds"].iloc[0] == 0.1 + 0.2
    assert back["odds"].iloc[1] == 1 / 3
    pd.testing.assert_frame_equal(back, table, check_exact=True)


def test_every_dtype_the_domains_use_survives_the_store():
    """Datetimes (naive and zoned), pandas strings, nullable ints, categories
    and a non-default index — the cycling result frame alone uses `string`."""
    table = pd.DataFrame(
        {"ds": pd.to_datetime(["2024-01-03 12:00:00.000000001", None]),
         "utc": pd.to_datetime(["2024-01-03", "2024-01-04"]).tz_localize("UTC"),
         "rider": pd.array(["Pogačar", None], dtype="string"),
         "rank": pd.array([1, None], dtype="Int64"),
         "terrain": pd.Categorical(["climb", "sprint"], categories=["climb", "sprint", "tt"]),
         "finished": [True, False]},
        index=pd.Index(["a", "b"], name="row"))

    back = jobs.decode_result(jobs.checked_encode({"table": table}))["table"]

    pd.testing.assert_frame_equal(back, table, check_exact=True)


def test_arrays_in_attrs_come_back_as_arrays_of_the_same_dtype():
    """`football/backtest.py:compare_models` keeps its held-out forecast
    matrices in attrs, and the calibration chart reads them back as arrays."""
    import numpy as np

    table = pd.DataFrame({"model": ["elo"]})
    table.attrs["forecasts"] = {"market": np.array([[0.5, 0.3, 0.2], [np.nan, 0.4, 0.6]]),
                                "outcomes": np.array([0, 2], dtype=np.int64)}

    back = jobs.decode_result(jobs.checked_encode({"table": table}))["table"].attrs["forecasts"]

    assert back["market"].dtype == np.float64 and back["market"].shape == (2, 3)
    assert np.isnan(back["market"][1, 0])
    assert back["outcomes"].dtype == np.int64


def test_a_frame_that_does_not_round_trip_is_refused():
    # Tuples come back as lists: equal-looking, not equal, and exactly the kind
    # of change nobody notices until a lookup by tuple key misses.
    table = pd.DataFrame({"key": [("tour", 1), ("tour", 2)]})

    with pytest.raises(jobs.JobResultError, match="does not come back"):
        jobs.checked_encode({"table": table})


def test_positional_column_names_are_refused_rather_than_renamed():
    with pytest.raises(jobs.JobResultError, match="columns must be strings"):
        jobs.checked_encode({"balls": pd.DataFrame({0: [1], 1: [2]})})


def test_a_value_json_cannot_hold_is_refused_by_name():
    with pytest.raises(jobs.JobResultError, match="Cannot store a object"):
        jobs.encode_result({"thing": object()})


def test_a_handler_must_return_a_mapping():
    with pytest.raises(jobs.JobResultError, match="must return a mapping"):
        jobs.encode_result([1, 2, 3])


# ------------------------------------------------------------ identity

def test_the_same_question_is_one_job(store):
    first = jobs.submit(store, "demo", {"scale": 2}, {"data": frame()})
    second = jobs.submit(store, "demo", {"scale": 2}, {"data": frame()})

    assert first["id"] == second["id"]
    assert jobs.counts(store)[jobs.QUEUED] == 1


def test_parameter_order_does_not_make_a_new_job(store):
    first = jobs.submit(store, "demo", {"a": 1, "b": 2}, {"data": frame()})
    second = jobs.submit(store, "demo", {"b": 2, "a": 1}, {"data": frame()})

    assert first["id"] == second["id"]


def test_one_corrected_cell_is_a_different_job(store):
    """A row count or a date range would call these the same data."""
    first = jobs.submit(store, "demo", {}, {"data": frame((1, 2, 3))})
    second = jobs.submit(store, "demo", {}, {"data": frame((1, 2, 4))})

    assert first["key"] != second["key"]
    assert first["id"] != second["id"]


def test_a_finished_job_is_the_answer_until_a_fresh_run_is_forced(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": echo}, once=True, log=lambda _: None)

    cached = jobs.submit(store, "demo", {}, {"data": frame()})
    fresh = jobs.submit(store, "demo", {}, {"data": frame()}, force=True)

    assert cached["id"] == job["id"] and cached["status"] == jobs.DONE
    assert fresh["id"] != job["id"] and fresh["status"] == jobs.QUEUED


def test_force_never_duplicates_a_job_that_is_already_in_flight(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})

    assert jobs.submit(store, "demo", {}, {"data": frame()}, force=True)["id"] == job["id"]


def test_a_failed_job_does_not_block_a_retry(store):
    failed = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": boom}, once=True, log=lambda _: None)

    retry = jobs.submit(store, "demo", {}, {"data": frame()})

    assert jobs.get(store, failed["id"])["status"] == jobs.FAILED
    assert retry["id"] != failed["id"]


def test_find_prefers_the_run_in_progress_over_the_last_answer(store):
    jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": echo}, once=True, log=lambda _: None)
    active = jobs.submit(store, "demo", {}, {"data": frame()}, force=True)

    assert jobs.find(store, active["key"])["id"] == active["id"]


def test_find_on_a_store_that_does_not_exist_yet_is_no_job(tmp_path):
    assert jobs.find(str(tmp_path / "absent.sqlite"), "key") is None
    assert jobs.list_jobs(str(tmp_path / "absent.sqlite")) == []
    assert jobs.live_workers(str(tmp_path / "absent.sqlite")) == []


# ------------------------------------------------------------ running

def test_a_worker_runs_the_job_on_the_inputs_stored_with_it(store):
    job = jobs.submit(store, "demo", {"scale": 3}, {"data": frame((1, 2, 3))}, label="demo run")

    assert jobs.run_worker(store, {"demo": echo}, once=True, log=lambda _: None) == 1

    done = jobs.get(store, job["id"])
    result = jobs.load_result(store, job["id"])
    assert done["status"] == jobs.DONE
    assert result["total"] == 18
    assert result["table"]["doubled"].tolist() == [2, 4, 6]
    # The worker records the code it started from on every job it runs.
    assert "git" in done["worker_manifest"]


def test_a_handler_that_raises_fails_the_job_with_its_traceback(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})

    jobs.run_worker(store, {"demo": boom}, once=True, log=lambda _: None)

    failed = jobs.get(store, job["id"])
    assert failed["status"] == jobs.FAILED
    assert "the model did not converge" in failed["error"]
    assert jobs.load_result(store, job["id"]) is None


def test_a_result_the_store_would_change_fails_the_job_rather_than_storing_it(store):
    def tuples(params, inputs, progress):
        return {"table": pd.DataFrame({"key": [("tour", 1)]})}

    job = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": tuples}, once=True, log=lambda _: None)

    assert jobs.get(store, job["id"])["status"] == jobs.FAILED
    assert "does not come back" in jobs.get(store, job["id"])["error"]


def test_a_worker_only_claims_kinds_it_can_run(store):
    """A worker without one domain's dependencies leaves its jobs for another."""
    jobs.submit(store, "cycling.eval", {}, {"data": frame()})

    assert jobs.run_worker(store, {"demo": echo}, once=True, log=lambda _: None) == 0
    assert jobs.counts(store)[jobs.QUEUED] == 1


def test_a_job_with_no_handler_in_the_claiming_worker_fails_by_name(store):
    job = jobs.submit(store, "mystery", {}, {"data": frame()})
    claimed = jobs.claim(store, "w1")

    assert jobs.run_job(store, claimed, {"demo": echo}, "w1") == jobs.FAILED
    assert "No handler for job kind 'mystery'" in jobs.get(store, job["id"])["error"]


def test_two_workers_cannot_claim_the_same_job(store):
    jobs.submit(store, "demo", {}, {"data": frame()})

    first = jobs.claim(store, "w1", ["demo"])
    second = jobs.claim(store, "w2", ["demo"])

    assert first is not None and first["worker"] == "w1"
    assert second is None


def test_concurrent_claims_hand_each_job_to_exactly_one_worker(store):
    for n in range(6):
        jobs.submit(store, "demo", {"n": n}, {"data": frame()})
    claimed, lock = [], threading.Lock()

    def worker(name):
        while (job := jobs.claim(store, name, ["demo"])) is not None:
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(claimed) == sorted(set(claimed))
    assert len(claimed) == 6


def test_progress_messages_are_recorded_on_the_job(store):
    seen = {}

    def reporting(params, inputs, progress):
        progress("window 3 of 10")
        seen.update(jobs.get(store, 1))
        return {"ok": True}

    jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": reporting}, once=True, log=lambda _: None)

    assert seen["progress"] == "window 3 of 10"


# ------------------------------------------------------------ dead and busy workers

def _age_heartbeat(store, job_id, seconds):
    stamp = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(timespec="microseconds")
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ?", (stamp, job_id))


def test_a_job_whose_worker_stopped_beating_goes_back_to_the_queue(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.claim(store, "dead-worker")
    _age_heartbeat(store, job["id"], 600)

    assert jobs.requeue_stale(store, stale_after=60) == [job["id"]]

    requeued = jobs.get(store, job["id"])
    assert requeued["status"] == jobs.QUEUED
    assert "stopped responding" in requeued["progress"]


def test_a_job_that_keeps_killing_its_worker_is_failed_not_retried_forever(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})
    for attempt in range(3):
        jobs.claim(store, f"w{attempt}")
        _age_heartbeat(store, job["id"], 600)
        jobs.requeue_stale(store, stale_after=60, max_attempts=3)

    final = jobs.get(store, job["id"])
    assert final["status"] == jobs.FAILED
    assert final["attempts"] == 3
    assert "not retried again" in final["error"]


def test_a_busy_worker_keeps_its_claim_through_a_long_silent_fit(store):
    """The handler reports nothing for longer than the stale limit. Without the
    heartbeat thread the job would read as orphaned and be handed to a second
    worker while the first was still fitting it."""
    touched = {}

    def slow(params, inputs, progress):
        threading.Event().wait(0.6)
        touched["ids"] = jobs.requeue_stale(store, stale_after=0.3)
        return {"ok": True}

    job = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": slow}, once=True, heartbeat_every=0.05, log=lambda _: None)

    assert touched["ids"] == []
    assert jobs.get(store, job["id"])["status"] == jobs.DONE


def test_a_worker_whose_job_was_requeued_cannot_overwrite_the_new_owner(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.claim(store, "slow-worker")
    _age_heartbeat(store, job["id"], 600)
    jobs.requeue_stale(store, stale_after=60)
    jobs.claim(store, "second-worker")

    jobs.finish(store, job["id"], "slow-worker", {"answer": "late"})

    assert jobs.get(store, job["id"])["status"] == jobs.RUNNING
    assert jobs.load_result(store, job["id"]) is None


def test_stopping_a_worker_mid_job_puts_the_job_back_without_spending_an_attempt(store):
    def interrupted(params, inputs, progress):
        raise KeyboardInterrupt

    job = jobs.submit(store, "demo", {}, {"data": frame()})
    with pytest.raises(KeyboardInterrupt):
        jobs.run_worker(store, {"demo": interrupted}, once=True, log=lambda _: None)

    released = jobs.get(store, job["id"])
    assert released["status"] == jobs.QUEUED
    assert released["attempts"] == 0
    assert jobs.live_workers(store) == []


def test_a_running_worker_is_listed_as_live_and_gone_after_it_stops(store):
    seen = {}

    def look(params, inputs, progress):
        seen["workers"] = jobs.live_workers(store)
        return {"ok": True}

    jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.run_worker(store, {"demo": look}, once=True, log=lambda _: None)

    assert len(seen["workers"]) == 1
    assert seen["workers"][0]["kinds"] == ["demo"]
    assert jobs.live_workers(store) == []


# ------------------------------------------------------------ run here

def test_running_in_process_stores_the_result_exactly_as_a_worker_would(store):
    messages = []

    job = jobs.run_here(store, "demo", {"scale": 2}, {"data": frame()}, echo,
                        progress=messages.append)

    assert job["status"] == jobs.DONE
    assert jobs.load_result(store, job["id"])["total"] == 12
    assert messages == ["summing"]
    assert job["worker"].startswith("inline:")


def test_running_in_process_again_reads_the_stored_answer_instead(store):
    calls = []

    def counted(params, inputs, progress):
        calls.append(1)
        return echo(params, inputs, progress)

    first = jobs.run_here(store, "demo", {}, {"data": frame()}, counted)
    second = jobs.run_here(store, "demo", {}, {"data": frame()}, counted)

    assert first["id"] == second["id"]
    assert len(calls) == 1


def test_running_in_process_does_not_steal_a_job_a_worker_already_has(store):
    job = jobs.submit(store, "demo", {}, {"data": frame()})
    jobs.claim(store, "w1")

    returned = jobs.run_here(store, "demo", {}, {"data": frame()}, echo)

    assert returned["id"] == job["id"]
    assert returned["worker"] == "w1"


# ------------------------------------------------------------ the store itself

def test_a_store_from_another_schema_version_is_refused(store):
    jobs.submit(store, "demo", {}, {"data": frame()})
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE _jobs_meta SET value = '0' WHERE key = 'schema_version'")

    with pytest.raises(jobs.JobError, match="schema 0"):
        jobs.list_jobs(store)


def test_unstorable_inputs_are_refused_before_anything_is_written(store):
    with pytest.raises(jobs.JobResultError, match="columns must be strings"):
        jobs.submit(store, "demo", {}, {"data": pd.DataFrame({0: [1]})})

    assert jobs.counts(store) == dict.fromkeys(jobs.STATUSES, 0)


def test_an_input_the_store_would_change_is_refused_before_it_is_queued(store):
    """A worker must compute on the frame the reader was looking at."""
    with pytest.raises(jobs.JobResultError, match="compute on different data"):
        jobs.submit(store, "demo", {}, {"data": pd.DataFrame({"key": [("tour", 1)]})})

    assert jobs.counts(store) == dict.fromkeys(jobs.STATUSES, 0)
