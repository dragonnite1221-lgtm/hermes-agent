from datetime import datetime, timedelta, timezone

import pytest

from cron import executions, jobs
from cron.scheduler_provider import InProcessCronScheduler
from tools.cronjob_tools import CRONJOB_SCHEMA


@pytest.fixture
def isolated_cron(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    return home


def _script_job(*, retry_interrupted=False):
    return jobs.create_job(
        prompt=None,
        schedule="every 1h",
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
        retry_interrupted=retry_interrupted,
    )


def test_retry_policy_is_explicit_and_restricted_to_no_agent_scripts(isolated_cron):
    default = _script_job()
    opted_in = _script_job(retry_interrupted=True)

    assert default["retry_interrupted"] is False
    assert opted_in["retry_interrupted"] is True
    with pytest.raises(ValueError, match="no_agent script jobs"):
        jobs.create_job(
            prompt="do work",
            schedule="every 1h",
            retry_interrupted=True,
        )
    assert "retry_interrupted" not in CRONJOB_SCHEMA["parameters"]["properties"]


def test_recurring_retry_waits_for_fire_claim_then_survives_due_scan(
    isolated_cron, monkeypatch
):
    now = datetime(2026, 8, 29, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    opted_in = _script_job(retry_interrupted=True)
    default = _script_job()
    future = (now + timedelta(hours=1)).isoformat()
    claim = {"at": now.isoformat(), "by": "old-owner"}
    jobs.update_job(opted_in["id"], {"next_run_at": future, "fire_claim": claim})
    jobs.update_job(default["id"], {"next_run_at": future})

    execution = executions.create_execution(opted_in["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    requeued = jobs.requeue_interrupted_jobs([
        execution,
        {"id": "exec-default", "job_id": default["id"]},
    ])

    assert requeued == {opted_in["id"]}
    recovered = jobs.get_job(opted_in["id"])
    eligible_at = now + timedelta(seconds=jobs.FIRE_CLAIM_TTL_SECONDS)
    assert recovered["next_run_at"] == eligible_at.isoformat()
    assert recovered["fire_claim"] == claim
    assert recovered["interrupted_retry"]["execution_ids"] == [execution["id"]]
    assert jobs.get_job(default["id"])["next_run_at"] == future

    monkeypatch.setattr(jobs, "_hermes_now", lambda: eligible_at + timedelta(seconds=1))
    due = jobs.get_due_jobs()
    assert [job["id"] for job in due] == [opted_in["id"]]
    jobs.advance_next_runs([opted_in["id"]])
    executions.mark_interrupted_executions_unknown([execution["id"]])
    claimed = jobs.claim_job_for_fire(opted_in["id"], return_job=True)
    assert isinstance(claimed, dict)
    assert "interrupted_retry" not in jobs.get_job(opted_in["id"])


def test_oneshot_retry_restores_one_dispatch_slot_exactly_once(
    isolated_cron, monkeypatch
):
    now = datetime(2026, 8, 29, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
        retry_interrupted=True,
    )
    claim = {"at": now.isoformat(), "by": "old-owner"}
    jobs.update_job(
        job["id"],
        {
            "next_run_at": now.isoformat(),
            "fire_claim": claim,
            "run_claim": {"at": now.isoformat(), "by": "dead-owner"},
            "repeat": {"times": 1, "completed": 1},
        },
    )
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])

    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    recovered = jobs.get_job(job["id"])
    assert recovered["repeat"] == {"times": 1, "completed": 0}
    assert recovered["run_claim"] is None

    eligible_at = now + timedelta(seconds=jobs.FIRE_CLAIM_TTL_SECONDS)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: eligible_at + timedelta(seconds=1))
    due = jobs.get_due_jobs()
    assert [candidate["id"] for candidate in due] == [job["id"]]
    executions.mark_interrupted_executions_unknown([execution["id"]])
    assert isinstance(jobs.claim_job_for_fire(job["id"], return_job=True), dict)
    assert jobs.claim_dispatch(job["id"]) is True
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1


def test_provider_recovery_requeues_before_marking_execution_unknown(
    isolated_cron, monkeypatch
):
    job = _script_job(retry_interrupted=True)
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    jobs.update_job(job["id"], {"next_run_at": future})
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started_at: False)

    assert InProcessCronScheduler().recover_interrupted() == 1

    recovered_execution = executions.latest_execution(job["id"])
    assert recovered_execution["status"] == "unknown"
    assert "retry was durably scheduled" in recovered_execution["error"]
    assert (
        datetime.fromisoformat(jobs.get_job(job["id"])["next_run_at"]).timestamp()
        <= datetime.now(timezone.utc).timestamp()
    )


def test_requeue_persistence_failure_leaves_execution_recoverable(
    isolated_cron, monkeypatch
):
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started_at: False)
    monkeypatch.setattr(
        jobs,
        "requeue_interrupted_jobs",
        lambda _job_ids: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    with pytest.raises(OSError, match="disk unavailable"):
        executions.recover_interrupted_executions()

    assert executions.latest_execution(job["id"])["status"] == "running"


def test_retry_cannot_fire_before_execution_ledger_handoff_commits(
    isolated_cron, monkeypatch
):
    now = datetime(2026, 8, 29, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])

    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    assert jobs.claim_job_for_fire(job["id"], return_job=True) is False
    pending = jobs.get_job(job["id"])
    assert pending["interrupted_retry"]["execution_ids"] == [execution["id"]]

    executions.mark_interrupted_executions_unknown([execution["id"]])
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    assert "interrupted_retry" not in jobs.get_job(job["id"])
