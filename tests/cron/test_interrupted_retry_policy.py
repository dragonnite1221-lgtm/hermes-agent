import contextlib
from datetime import datetime, timedelta, timezone

import pytest

from cron import executions, jobs
from cron.scheduler_provider import FireClaimNotAcquiredError, InProcessCronScheduler
from tools.cronjob_tools import CRONJOB_SCHEMA, _format_job


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
    assert _format_job(default)["retry_interrupted"] is False
    assert _format_job(opted_in)["retry_interrupted"] is True
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

    overdue_retry_at = eligible_at + timedelta(hours=2)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: overdue_retry_at)
    due_before_handoff = jobs.get_due_jobs()
    assert opted_in["id"] not in {candidate["id"] for candidate in due_before_handoff}
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={opted_in["id"]}
    )
    due = jobs.get_due_jobs()
    assert opted_in["id"] in {candidate["id"] for candidate in due}
    retry_due_at = jobs.get_job(opted_in["id"])["next_run_at"]
    assert retry_due_at == eligible_at.isoformat()
    jobs.advance_next_runs([opted_in["id"]])
    assert jobs.get_job(opted_in["id"])["next_run_at"] == retry_due_at
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
    assert jobs.get_due_jobs() == []
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    due = jobs.get_due_jobs()
    assert [candidate["id"] for candidate in due] == [job["id"]]
    assert isinstance(jobs.claim_job_for_fire(job["id"], return_job=True), dict)
    assert jobs.claim_dispatch(job["id"]) is True
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1


def test_paused_oneshot_retry_resumes_from_retry_eligibility(
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
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    eligible_at = jobs.get_job(job["id"])["interrupted_retry"]["eligible_at"]

    assert jobs.pause_job(job["id"])["state"] == "paused"
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now + timedelta(hours=2))
    resumed = jobs.resume_job(job["id"])

    assert resumed["enabled"] is True
    assert resumed["state"] == "scheduled"
    assert resumed["next_run_at"] == eligible_at
    assert resumed["interrupted_retry"]["execution_ids"] == [execution["id"]]


def test_resume_and_requeue_commit_under_one_jobs_lock(
    isolated_cron, monkeypatch
):
    import threading

    job = _script_job(retry_interrupted=True)
    jobs.pause_job(job["id"])
    execution = {"id": "exec-resume-race", "job_id": job["id"]}
    resume_loaded = threading.Event()
    allow_resume = threading.Event()
    requeue_done = threading.Event()
    real_load_jobs = jobs.load_jobs
    resume_thread = None

    def blocking_load_jobs():
        records = real_load_jobs()
        if threading.current_thread() is resume_thread and not resume_loaded.is_set():
            resume_loaded.set()
            assert allow_resume.wait(timeout=5)
        return records

    monkeypatch.setattr(jobs, "load_jobs", blocking_load_jobs)
    resume_thread = threading.Thread(target=jobs.resume_job, args=(job["id"],))
    resume_thread.start()
    assert resume_loaded.wait(timeout=5)

    def requeue():
        jobs.requeue_interrupted_jobs([execution])
        requeue_done.set()

    requeue_thread = threading.Thread(target=requeue)
    requeue_thread.start()
    assert requeue_done.wait(timeout=0.1) is False
    allow_resume.set()
    resume_thread.join(timeout=5)
    requeue_thread.join(timeout=5)

    assert not resume_thread.is_alive()
    assert not requeue_thread.is_alive()
    recovered = jobs.get_job(job["id"])
    assert recovered["interrupted_retry"]["execution_ids"] == [execution["id"]]
    assert recovered["next_run_at"] == recovered["interrupted_retry"]["eligible_at"]


def test_resume_and_retry_cancel_commit_under_one_jobs_lock(
    isolated_cron, monkeypatch
):
    import threading

    now = datetime.now(timezone.utc)
    job = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
        retry_interrupted=True,
    )
    jobs.update_job(job["id"], {"repeat": {"times": 1, "completed": 1}})
    execution = {"id": "exec-resume-cancel-race", "job_id": job["id"]}
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    jobs.pause_job(job["id"])
    resume_loaded = threading.Event()
    allow_resume = threading.Event()
    cancel_done = threading.Event()
    real_load_jobs = jobs.load_jobs
    resume_thread = None

    def blocking_load_jobs():
        records = real_load_jobs()
        if threading.current_thread() is resume_thread and not resume_loaded.is_set():
            resume_loaded.set()
            assert allow_resume.wait(timeout=5)
        return records

    monkeypatch.setattr(jobs, "load_jobs", blocking_load_jobs)
    resume_thread = threading.Thread(target=jobs.resume_job, args=(job["id"],))
    resume_thread.start()
    assert resume_loaded.wait(timeout=5)

    def cancel_retry():
        jobs.update_job(job["id"], {"retry_interrupted": False})
        cancel_done.set()

    cancel_thread = threading.Thread(target=cancel_retry)
    cancel_thread.start()
    assert cancel_done.wait(timeout=0.1) is False
    allow_resume.set()
    resume_thread.join(timeout=5)
    cancel_thread.join(timeout=5)

    assert not resume_thread.is_alive()
    assert not cancel_thread.is_alive()
    cancelled = jobs.get_job(job["id"])
    assert "interrupted_retry" not in cancelled
    assert cancelled["retry_interrupted"] is False
    assert cancelled["enabled"] is False
    assert cancelled["state"] == "error"
    assert cancelled["next_run_at"] is None


def test_recovery_records_retry_for_job_paused_before_restart(
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
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    paused = jobs.pause_job(job["id"], reason="operator maintenance")
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started_at: False)

    assert executions.recover_interrupted_executions() == 1

    recovered = jobs.get_job(job["id"])
    assert recovered["state"] == "paused"
    assert recovered["enabled"] is False
    assert recovered["paused_at"] == paused["paused_at"]
    assert recovered["interrupted_retry"]["execution_ids"] == [execution["id"]]
    assert executions.latest_execution(job["id"])["status"] == "unknown"
    assert jobs.get_due_jobs() == []

    resumed = jobs.resume_job(job["id"])
    assert resumed["state"] == "scheduled"
    assert resumed["next_run_at"] == recovered["interrupted_retry"]["eligible_at"]


def test_malformed_schedule_does_not_abort_interrupted_recovery(isolated_cron):
    malformed = _script_job(retry_interrupted=True)
    healthy = _script_job(retry_interrupted=True)
    with jobs._jobs_lock():
        stored = jobs.load_jobs()
        for record in stored:
            if record["id"] == malformed["id"]:
                record["schedule"] = None
        jobs.save_jobs(stored)

    malformed_execution = executions.create_execution(
        malformed["id"], source="builtin"
    )
    healthy_execution = executions.create_execution(healthy["id"], source="builtin")
    executions.mark_execution_running(malformed_execution["id"])
    executions.mark_execution_running(healthy_execution["id"])

    assert jobs.requeue_interrupted_jobs(
        [malformed_execution, healthy_execution]
    ) == {malformed["id"], healthy["id"]}
    assert jobs.get_job(malformed["id"])["interrupted_retry"]["execution_ids"] == [
        malformed_execution["id"]
    ]
    assert jobs.get_job(healthy["id"])["interrupted_retry"]["execution_ids"] == [
        healthy_execution["id"]
    ]


def test_malformed_schedule_does_not_block_retry_policy_update(isolated_cron):
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}

    with jobs._jobs_lock():
        stored = jobs.load_jobs()
        stored[0]["schedule"] = None
        jobs.save_jobs(stored)

    updated = jobs.update_job(job["id"], {"retry_interrupted": False})

    assert updated["retry_interrupted"] is False
    assert "interrupted_retry" not in updated
    assert updated["schedule"] == {}
    assert updated["next_run_at"] is None


def test_malformed_retry_ids_do_not_abort_healthy_marker_reconciliation(
    isolated_cron,
):
    malformed = _script_job(retry_interrupted=True)
    healthy = _script_job(retry_interrupted=True)
    malformed_execution = executions.create_execution(
        malformed["id"], source="builtin"
    )
    healthy_execution = executions.create_execution(healthy["id"], source="builtin")
    assert jobs.requeue_interrupted_jobs(
        [malformed_execution, healthy_execution]
    ) == {malformed["id"], healthy["id"]}
    executions.finish_execution(healthy_execution["id"], success=True)

    with jobs._jobs_lock():
        stored = jobs.load_jobs()
        for record in stored:
            if record["id"] == malformed["id"]:
                record["interrupted_retry"]["execution_ids"] = 42
        jobs.save_jobs(stored)

    assert jobs.reconcile_interrupted_retry_markers() == {healthy["id"]}
    assert "interrupted_retry" in jobs.get_job(malformed["id"])
    assert "interrupted_retry" not in jobs.get_job(healthy["id"])


@pytest.mark.parametrize(
    "repeat",
    [
        {"times": "1", "completed": 1},
        {"times": 1, "completed": None},
        {"times": 1, "completed": -1},
    ],
)
def test_malformed_repeat_skips_only_that_interrupted_job(isolated_cron, repeat):
    now = datetime.now(timezone.utc)
    malformed = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
        retry_interrupted=True,
    )
    healthy = _script_job(retry_interrupted=True)
    with jobs._jobs_lock():
        stored = jobs.load_jobs()
        for record in stored:
            if record["id"] == malformed["id"]:
                record["repeat"] = repeat
        jobs.save_jobs(stored)
    malformed_execution = executions.create_execution(
        malformed["id"], source="builtin"
    )
    healthy_execution = executions.create_execution(healthy["id"], source="builtin")

    assert jobs.requeue_interrupted_jobs(
        [malformed_execution, healthy_execution]
    ) == {healthy["id"]}
    assert "interrupted_retry" not in jobs.get_job(malformed["id"])
    assert jobs.get_job(healthy["id"])["interrupted_retry"]["execution_ids"] == [
        healthy_execution["id"]
    ]


def test_schedule_edit_preserves_pending_retry_eligibility(isolated_cron):
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    eligible_at = jobs.get_job(job["id"])["interrupted_retry"]["eligible_at"]

    updated = jobs.update_job(job["id"], {"schedule": "every 24h"})

    assert updated["schedule"]["kind"] == "interval"
    assert updated["next_run_at"] == eligible_at
    assert updated["interrupted_retry"]["execution_ids"] == [execution["id"]]


def test_dead_fire_claim_loser_is_discarded_without_retry(isolated_cron, monkeypatch):
    job = _script_job(retry_interrupted=True)
    loser = executions.create_execution(
        job["id"], source="external", fire_claim_acquired=False
    )
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-process' WHERE id=?",
            (loser["id"],),
        )
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution(job["id"]) is None
    assert "interrupted_retry" not in jobs.get_job(job["id"])


def test_dead_fire_claim_winner_is_promoted_and_recovered(
    isolated_cron, monkeypatch
):
    job = _script_job(retry_interrupted=True)
    winner = executions.create_execution(
        job["id"], source="external", fire_claim_acquired=False
    )
    assert isinstance(
        jobs.claim_job_for_fire(
            job["id"], return_job=True, execution_id=winner["id"]
        ),
        dict,
    )
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='dead-process' WHERE id=?",
            (winner["id"],),
        )
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)

    assert executions.recover_interrupted_executions() == 1
    assert executions.latest_execution(job["id"])["status"] == "unknown"
    assert jobs.get_job(job["id"])["interrupted_retry"]["execution_ids"] == [
        winner["id"]
    ]


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
        lambda _job_ids, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
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

    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)


def test_recovery_heartbeat_race_cancels_retry_and_restores_schedule(
    isolated_cron, monkeypatch
):
    job = _script_job(retry_interrupted=True)
    before_next_run = jobs.get_job(job["id"])["next_run_at"]
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    real_requeue = jobs.requeue_interrupted_jobs

    def renew_after_requeue(candidates, **kwargs):
        result = real_requeue(candidates, **kwargs)
        assert executions.heartbeat_execution(execution["id"]) is True
        return result

    monkeypatch.setattr(jobs, "requeue_interrupted_jobs", renew_after_requeue)

    assert executions.recover_interrupted_executions() == 0
    restored = jobs.get_job(job["id"])
    assert "interrupted_retry" not in restored
    assert restored["next_run_at"] == before_next_run
    assert executions.latest_execution(job["id"])["status"] == "running"


def test_recovery_waits_for_side_effect_fence_while_owner_renews(
    isolated_cron, monkeypatch
):
    import threading

    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job(retry_interrupted=True)
    claimed = claim_fire_with_execution(job["id"], source="direct")
    assert isinstance(claimed, dict)
    execution_id = claimed["execution_id"]
    executions.mark_execution_running(execution_id)
    owner = claimed["fire_claim"]["by"]
    original_at = claimed["fire_claim"]["at"]
    candidate = executions.latest_execution(job["id"])
    candidate["_fire_claim_generation"] = f"{owner}:{original_at}"
    monkeypatch.setattr(
        executions, "interrupted_execution_candidates", lambda: [candidate]
    )

    real_recovery_fence = jobs.fire_recovery_fence
    recovery_waiting = threading.Event()

    @contextlib.contextmanager
    def observed_recovery_fence(job_id):
        recovery_waiting.set()
        with real_recovery_fence(job_id) as acquired:
            yield acquired

    monkeypatch.setattr(jobs, "fire_recovery_fence", observed_recovery_fence)
    bumped = datetime.fromisoformat(original_at) + timedelta(seconds=1)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: bumped)
    recovery_result = []
    heartbeat_result = []

    with jobs.fire_claim_fence(job["id"], expected_owner=owner) as owns_claim:
        assert owns_claim is True
        recovery_thread = threading.Thread(
            target=lambda: recovery_result.append(
                executions.recover_interrupted_executions()
            )
        )
        recovery_thread.start()
        assert recovery_waiting.wait(timeout=5)

        heartbeat_thread = threading.Thread(
            target=lambda: heartbeat_result.append(
                jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)
            )
        )
        heartbeat_thread.start()
        heartbeat_thread.join(timeout=2)
        assert not heartbeat_thread.is_alive()
        assert heartbeat_result == [True]
        assert recovery_thread.is_alive()

    recovery_thread.join(timeout=5)
    assert not recovery_thread.is_alive()
    assert recovery_result == [0]
    assert executions.latest_execution(job["id"])["status"] == "running"
    assert "interrupted_retry" not in jobs.get_job(job["id"])


def test_ambiguous_abort_restores_occurrence_during_total_ledger_outage(
    isolated_cron, monkeypatch
):
    from cron.scheduler_provider import (
        abort_fire_claim_execution,
        claim_fire_with_execution,
    )

    job = _script_job(retry_interrupted=True)
    claimed = claim_fire_with_execution(job["id"], source="direct")
    assert isinstance(claimed, dict)
    execution_id = claimed["execution_id"]
    real_transaction = executions._transaction

    def ledger_unavailable():
        raise OSError("executions ledger unavailable")

    monkeypatch.setattr(executions, "_transaction", ledger_unavailable)
    assert abort_fire_claim_execution(
        claimed,
        "ambiguous dispatch outcome",
        prefer_recovery=True,
    ) == "restored"
    monkeypatch.setattr(executions, "_transaction", real_transaction)

    restored = jobs.get_job(job["id"])
    assert restored["fire_claim"] is None
    assert restored["_pre_run_abort_execution_ids"] == [execution_id]
    assert executions.recover_interrupted_executions() == 0
    recovered = executions.latest_execution(job["id"])
    assert recovered["status"] == "failed"
    assert "_pre_run_abort_execution_ids" not in jobs.get_job(job["id"])
    assert "interrupted_retry" not in jobs.get_job(job["id"])


def test_restored_pre_run_abort_tombstone_prevents_late_duplicate_retry(
    isolated_cron, monkeypatch
):
    from cron.scheduler_provider import (
        abort_fire_claim_execution,
        claim_fire_with_execution,
    )

    job = _script_job(retry_interrupted=True)
    before = jobs.get_job(job["id"])
    claimed = claim_fire_with_execution(job["id"], source="direct")
    assert isinstance(claimed, dict)
    execution_id = claimed["execution_id"]
    real_finish_execution = executions.finish_execution
    finish_attempts = []

    def ledger_unavailable(execution_id, **_kwargs):
        finish_attempts.append(execution_id)
        raise OSError("executions ledger unavailable")

    monkeypatch.setattr(executions, "finish_execution", ledger_unavailable)
    assert abort_fire_claim_execution(claimed, "pre-run validation failed") == "restored"
    assert finish_attempts == [execution_id] * 3
    restored = jobs.get_job(job["id"])
    assert restored["next_run_at"] == before["next_run_at"]
    assert restored["fire_claim"] is None
    assert restored["_pre_run_abort_execution_ids"] == [execution_id]

    monkeypatch.setattr(executions, "finish_execution", real_finish_execution)
    assert executions.recover_interrupted_executions() == 0
    terminal = executions.latest_execution(job["id"])
    assert terminal["id"] == execution_id
    assert terminal["status"] == "failed"
    recovered_job = jobs.get_job(job["id"])
    assert "_pre_run_abort_execution_ids" not in recovered_job
    assert "interrupted_retry" not in recovered_job

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-process")
    assert executions.recover_interrupted_executions() == 0
    assert "interrupted_retry" not in jobs.get_job(job["id"])


def test_clean_pre_run_rollback_releases_exact_oneshot_due_claim(
    isolated_cron, monkeypatch
):
    from cron.scheduler_provider import (
        abort_fire_claim_execution,
        claim_fire_with_execution,
    )

    now = datetime(2026, 8, 30, 2, 0, tzinfo=timezone.utc)
    clock = [now]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    job = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
    )
    clock[0] = now + timedelta(minutes=2)
    assert [candidate["id"] for candidate in jobs.get_due_jobs()] == [job["id"]]
    due_claim = jobs.get_job(job["id"])["run_claim"]
    claimed = claim_fire_with_execution(job["id"], source="builtin")
    assert isinstance(claimed, dict)

    assert abort_fire_claim_execution(claimed, "pre-run validation failed") == "restored"
    restored = jobs.get_job(job["id"])
    assert restored["fire_claim"] is None
    assert restored["run_claim"] is None

    clock[0] += timedelta(seconds=1)
    assert [candidate["id"] for candidate in jobs.get_due_jobs()] == [job["id"]]
    assert jobs.get_job(job["id"])["run_claim"]["at"] != due_claim["at"]


def test_ambiguous_oneshot_dispatch_is_exactly_rolled_back_before_retry(
    isolated_cron, monkeypatch
):
    from cron.scheduler_provider import abort_fire_claim_execution
    from plugins.cron_providers.chronos import ChronosCronScheduler

    now = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
    )
    before = jobs.get_job(job["id"])
    claimed = ChronosCronScheduler().claim_fire(job["id"])
    assert isinstance(claimed, dict)
    assert claimed["_aborted_fire_rearm_delay_seconds"] == 5

    # Model a jobs-store save that committed the finite dispatch increment but
    # raised before claim_dispatch() could acknowledge it to the caller.
    assert jobs.claim_dispatch(job["id"]) is True
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1

    assert abort_fire_claim_execution(
        claimed,
        "ambiguous dispatch outcome",
        prefer_recovery=True,
    ) == "restored"
    restored = jobs.get_job(job["id"])
    assert restored["fire_claim"] is None
    assert restored["repeat"] == before["repeat"]
    assert restored["next_run_at"] == (
        now + timedelta(minutes=1, microseconds=1)
    ).isoformat()
    assert "_pre_run_abort_execution_ids" not in restored
    assert executions.latest_execution(job["id"])["status"] == "failed"


def test_concurrent_recovery_winner_preserves_interrupted_retry(
    isolated_cron, monkeypatch
):
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    real_mark_unknown = executions.mark_interrupted_executions_unknown

    def concurrent_winner(execution_ids, **kwargs):
        assert real_mark_unknown(execution_ids, **kwargs)
        return []

    monkeypatch.setattr(
        executions, "mark_interrupted_executions_unknown", concurrent_winner
    )

    assert executions.recover_interrupted_executions() == 0
    recovered = jobs.get_job(job["id"])
    assert recovered["interrupted_retry"]["execution_ids"] == [execution["id"]]
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is True
    assert executions.latest_execution(job["id"])["status"] == "unknown"


def test_recovery_crash_marker_is_reconciled_after_owner_renews(
    isolated_cron,
):
    job = _script_job(retry_interrupted=True)
    before_next_run = jobs.get_job(job["id"])["next_run_at"]
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])

    # Simulate recovery dying after jobs.json requeue but before its SQLite CAS.
    candidate = executions.latest_execution(job["id"])
    assert jobs.requeue_interrupted_jobs(
        [candidate], expected_heartbeats={candidate["id"]: candidate["heartbeat_at"]}
    ) == {job["id"]}
    assert executions.heartbeat_execution(execution["id"]) is True

    assert jobs.reconcile_interrupted_retry_markers() == {job["id"]}
    restored = jobs.get_job(job["id"])
    assert "interrupted_retry" not in restored
    assert restored["next_run_at"] == before_next_run
    assert "interrupted_retry" not in jobs.get_job(job["id"])


def test_builtin_due_scan_waits_for_handoff_and_consumes_retry_on_completion(
    isolated_cron, monkeypatch
):
    now = datetime(2026, 8, 29, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])

    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    assert jobs.get_due_jobs() == []

    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    due = jobs.get_due_jobs()
    assert [candidate["id"] for candidate in due] == [job["id"]]
    assert "interrupted_retry" in jobs.get_job(job["id"])

    assert jobs.mark_job_run(job["id"], success=True) is True

    assert "interrupted_retry" not in jobs.get_job(job["id"])
    assert executions.execution_ids_are_terminal(
        [abandoned["id"]], job_id=job["id"]
    ) is False


def test_lost_fire_claim_does_not_create_phantom_failed_execution(isolated_cron):
    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job()
    assert isinstance(jobs.claim_job_for_fire(job["id"], return_job=True), dict)

    assert claim_fire_with_execution(job["id"], source="external") is None
    assert executions.list_executions(job_id=job["id"]) == []


def test_post_cas_execution_fence_retries_idempotently(isolated_cron, monkeypatch):
    import cron.scheduler_provider as provider

    job = _script_job()
    original_mark = executions.mark_fire_claim_acquired
    attempts = []

    def flaky_mark(execution_id):
        attempts.append(execution_id)
        if len(attempts) < 3:
            raise OSError("execution ledger temporarily unavailable")
        return original_mark(execution_id)

    monkeypatch.setattr(executions, "mark_fire_claim_acquired", flaky_mark)

    claimed = provider.claim_fire_with_execution(job["id"], source="external")

    assert claimed is not None
    assert attempts == [claimed["execution_id"]] * 3
    assert executions.latest_execution(job["id"])["fire_claim_acquired"] == 1
    assert jobs.get_job(job["id"])["fire_claim"]["execution_id"] == claimed["execution_id"]


def test_post_cas_execution_fence_failure_restores_interrupted_retry(
    isolated_cron, monkeypatch
):
    import cron.scheduler_provider as provider

    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    before = jobs.get_job(job["id"])

    monkeypatch.setattr(
        executions,
        "mark_fire_claim_acquired",
        lambda _execution_id: (_ for _ in ()).throw(OSError("ledger offline")),
    )

    with pytest.raises(FireClaimNotAcquiredError, match="occurrence was restored"):
        provider.claim_fire_with_execution(job["id"], source="external")

    restored = jobs.get_job(job["id"])
    assert restored["fire_claim"] is None
    assert restored["next_run_at"] == before["next_run_at"]
    assert restored["interrupted_retry"] == before["interrupted_retry"]
    assert executions.execution_ids_are_terminal(
        [abandoned["id"]], job_id=job["id"]
    ) is True
    assert all(
        row["fire_claim_acquired"] == 1
        for row in executions.list_executions(job_id=job["id"])
    )


def test_total_fence_and_rollback_failure_recovers_provisional_winner_in_process(
    isolated_cron, monkeypatch
):
    import cron.scheduler_provider as provider

    job = _script_job(retry_interrupted=True)
    real_fire_job_lock = jobs._fire_job_lock
    lock_calls = 0

    @contextlib.contextmanager
    def rollback_unavailable_fire_job_lock(job_id):
        nonlocal lock_calls
        lock_calls += 1
        if lock_calls == 1:
            with real_fire_job_lock(job_id) as acquired:
                yield acquired
            return
        yield False

    monkeypatch.setattr(
        executions,
        "mark_fire_claim_acquired",
        lambda _execution_id: (_ for _ in ()).throw(OSError("ledger offline")),
    )
    monkeypatch.setattr(jobs, "_fire_job_lock", rollback_unavailable_fire_job_lock)

    with pytest.raises(FireClaimNotAcquiredError, match="rollback also failed"):
        provider.claim_fire_with_execution(job["id"], source="external")

    provisional = executions.latest_execution(job["id"])
    assert provisional["fire_claim_acquired"] == 0
    assert jobs.get_job(job["id"])["fire_claim"]["execution_id"] == provisional["id"]

    monkeypatch.setattr(jobs, "_fire_job_lock", real_fire_job_lock)
    assert executions.recover_interrupted_executions() == 1
    recovered = executions.latest_execution(job["id"])
    assert recovered["id"] == provisional["id"]
    assert recovered["status"] == "unknown"
    assert jobs.get_job(job["id"])["interrupted_retry"]["execution_ids"] == [
        provisional["id"]
    ]


def test_unacquired_reconciliation_revalidates_winner_under_fire_fence(
    isolated_cron, monkeypatch
):
    job = _script_job()
    old = executions.create_execution(
        job["id"], source="external", fire_claim_acquired=False
    )
    replacement = executions.create_execution(
        job["id"], source="external", fire_claim_acquired=False
    )
    claimed_at = datetime.now(timezone.utc).isoformat()
    jobs.update_job(
        job["id"],
        {
            "fire_claim": {
                "by": "old-owner",
                "at": claimed_at,
                "execution_id": old["id"],
            }
        },
    )
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    real_recovery_fence = jobs.fire_recovery_fence

    @contextlib.contextmanager
    def replace_before_revalidation(job_id):
        with real_recovery_fence(job_id) as acquired:
            if acquired:
                with jobs._jobs_lock():
                    stored = jobs.load_jobs()
                    for record in stored:
                        if record["id"] == job_id:
                            record["fire_claim"] = {
                                "by": "replacement-owner",
                                "at": claimed_at,
                                "execution_id": replacement["id"],
                            }
                    jobs.save_jobs(stored)
            yield acquired

    monkeypatch.setattr(jobs, "fire_recovery_fence", replace_before_revalidation)

    assert executions.reconcile_unacquired_executions() == (1, 1)
    assert executions.latest_execution(job["id"])["id"] == replacement["id"]
    assert executions.latest_execution(job["id"])["fire_claim_acquired"] == 1
    assert all(
        record["id"] != old["id"]
        for record in executions.list_executions(job_id=job["id"])
    )


def test_execution_fence_confirmation_is_idempotent(isolated_cron):
    job = _script_job()
    execution = executions.create_execution(
        job["id"], source="external", fire_claim_acquired=False
    )

    first = executions.mark_fire_claim_acquired(execution["id"])
    second = executions.mark_fire_claim_acquired(execution["id"])

    assert first is not None
    assert second is not None
    assert first["id"] == second["id"] == execution["id"]


def test_failed_forced_fire_restores_paused_state(isolated_cron, monkeypatch):
    import cron.scheduler_provider as provider

    job = _script_job()
    before = jobs.pause_job(job["id"], reason="operator maintenance")
    monkeypatch.setattr(
        executions,
        "mark_fire_claim_acquired",
        lambda _execution_id: None,
    )

    with pytest.raises(FireClaimNotAcquiredError, match="occurrence was restored"):
        provider.claim_fire_with_execution(
            job["id"], source="manual", force=True
        )

    restored = jobs.get_job(job["id"])
    assert restored["enabled"] == before["enabled"] is False
    assert restored["state"] == before["state"] == "paused"
    assert restored["paused_at"] == before["paused_at"]
    assert restored["paused_reason"] == before["paused_reason"]
    assert restored["next_run_at"] == before["next_run_at"]
    assert restored["fire_claim"] is None


def test_fence_rollback_preserves_concurrent_operator_pause(
    isolated_cron, monkeypatch
):
    import cron.scheduler_provider as provider

    job = _script_job()
    pause_results = []

    def fail_after_pause(_execution_id):
        if not pause_results:
            pause_results.append(
                jobs.pause_job(job["id"], reason="operator won the race")
            )
        return None

    monkeypatch.setattr(executions, "mark_fire_claim_acquired", fail_after_pause)

    with pytest.raises(FireClaimNotAcquiredError, match="occurrence was restored"):
        provider.claim_fire_with_execution(job["id"], source="external")

    stored = jobs.get_job(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "paused"
    assert stored["paused_reason"] == "operator won the race"
    assert stored["paused_at"] == pause_results[0]["paused_at"]
    assert stored["fire_claim"] is None


def test_fence_rollback_does_not_resurrect_concurrently_disabled_retry(
    isolated_cron, monkeypatch
):
    import cron.scheduler_provider as provider

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    updates = []

    def fail_after_policy_disable(_execution_id):
        if not updates:
            updates.append(
                jobs.update_job(job["id"], {"retry_interrupted": False})
            )
        return None

    monkeypatch.setattr(
        executions, "mark_fire_claim_acquired", fail_after_policy_disable
    )

    with pytest.raises(FireClaimNotAcquiredError, match="occurrence was restored"):
        provider.claim_fire_with_execution(job["id"], source="external")

    stored = jobs.get_job(job["id"])
    assert stored["retry_interrupted"] is False
    assert "interrupted_retry" not in stored
    assert stored["fire_claim"] is None
    assert stored["next_run_at"] == updates[0]["next_run_at"]
    assert executions.execution_ids_are_terminal(
        [abandoned["id"]], job_id=job["id"]
    ) is False


def test_fence_rollback_recreates_ack_pruned_during_claim_gap(
    isolated_cron, monkeypatch
):
    import cron.scheduler_provider as provider

    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    assert executions.execution_ids_are_terminal(
        [abandoned["id"]], job_id=job["id"]
    ) is True
    pruned = []

    def fail_after_orphan_prune(_execution_id):
        if not pruned:
            pruned.append(executions.prune_orphaned_interrupted_retry_acks())
        return None

    monkeypatch.setattr(executions, "mark_fire_claim_acquired", fail_after_orphan_prune)

    with pytest.raises(FireClaimNotAcquiredError, match="occurrence was restored"):
        provider.claim_fire_with_execution(job["id"], source="external")

    restored = jobs.get_job(job["id"])
    assert pruned == [1]
    assert restored["interrupted_retry"]["execution_ids"] == [abandoned["id"]]
    assert executions.execution_ids_are_terminal(
        [abandoned["id"]], job_id=job["id"]
    ) is True


def test_ledger_create_failure_never_attempts_or_consumes_fire_claim(
    isolated_cron, monkeypatch
):
    import cron.scheduler_provider as provider

    attempted_claims = []
    monkeypatch.setattr(
        executions,
        "create_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda *_args, **_kwargs: attempted_claims.append(True),
    )

    with pytest.raises(FireClaimNotAcquiredError, match="durable execution"):
        provider.claim_fire_with_execution("job-never-claimed", source="direct")
    assert attempted_claims == []


def test_manual_claim_setup_failure_preserves_interrupted_retry(
    isolated_cron, monkeypatch
):
    import tools.cronjob_tools as cronjob_tools

    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    before = jobs.get_job(job["id"])["interrupted_retry"]
    marked = []

    def fail_setup(*_args, **_kwargs):
        raise FireClaimNotAcquiredError("ledger unavailable")

    monkeypatch.setattr(cronjob_tools, "claim_job_for_fire", fail_setup)
    monkeypatch.setattr(
        cronjob_tools,
        "mark_job_run",
        lambda *_args, **_kwargs: marked.append(True),
    )

    immediate = cronjob_tools._execute_job_now(job)
    assert immediate["claimed"] is False
    assert marked == []
    assert jobs.get_job(job["id"])["interrupted_retry"] == before

    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    background = cronjob_tools._try_dispatch_background_run(
        job, session_id="test-session"
    )
    assert background["claimed"] is False
    assert background["dispatched"] is False
    assert marked == []
    assert jobs.get_job(job["id"])["interrupted_retry"] == before


def test_terminal_retry_ack_survives_execution_pruning(isolated_cron, monkeypatch):
    now = datetime(2026, 8, 29, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])

    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    recovered = executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    assert [row["id"] for row in recovered] == [execution["id"]]
    assert executions.latest_execution(job["id"]) is None
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is True
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id="different-job"
    ) is False
    assert isinstance(jobs.claim_job_for_fire(job["id"], return_job=True), dict)
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is False


def test_disabling_retry_policy_cancels_pending_recurring_retry(
    isolated_cron, monkeypatch
):
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is True

    updated = jobs.update_job(job["id"], {"retry_interrupted": False})

    assert updated["retry_interrupted"] is False
    assert "interrupted_retry" not in updated
    assert datetime.fromisoformat(updated["next_run_at"]) > datetime.now(timezone.utc)
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is False


def test_disabling_retry_policy_terminalizes_pending_oneshot(isolated_cron):
    now = datetime.now(timezone.utc)
    job = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
        retry_interrupted=True,
    )
    jobs.update_job(job["id"], {"repeat": {"times": 1, "completed": 1}})
    execution = {"id": "exec-once-cancel", "job_id": job["id"]}
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}

    updated = jobs.update_job(job["id"], {"retry_interrupted": False})

    assert updated["repeat"] == {"times": 1, "completed": 1}
    assert updated["enabled"] is False
    assert updated["state"] == "error"
    assert updated["last_status"] == "unknown"
    assert updated["next_run_at"] is None
    assert "interrupted_retry" not in updated


def test_replacing_pending_oneshot_while_disabling_retry_keeps_new_schedule(
    isolated_cron,
):
    now = datetime.now(timezone.utc)
    job = jobs.create_job(
        prompt=None,
        schedule=(now + timedelta(minutes=1)).isoformat(),
        script="idempotent-sync.py",
        no_agent=True,
        deliver="local",
        retry_interrupted=True,
    )
    jobs.update_job(job["id"], {"repeat": {"times": 1, "completed": 1}})
    execution = {"id": "exec-once-replaced", "job_id": job["id"]}
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    replacement = now + timedelta(hours=2)

    updated = jobs.update_job(
        job["id"],
        {"schedule": replacement.isoformat(), "retry_interrupted": False},
    )

    assert updated["retry_interrupted"] is False
    assert "interrupted_retry" not in updated
    assert updated["enabled"] is True
    assert updated["state"] == "scheduled"
    assert updated["repeat"] == {"times": 1, "completed": 0}
    assert datetime.fromisoformat(updated["next_run_at"]) == replacement


@pytest.mark.parametrize(
    ("no_agent", "script"),
    [(False, "unsafe-agent.py"), ("yes", "unsafe-agent.py"), (True, None)],
)
def test_recovery_revalidates_hand_edited_retry_policy(
    isolated_cron, no_agent, script
):
    job = _script_job()
    with jobs._jobs_lock():
        stored = jobs.load_jobs()
        stored[0]["retry_interrupted"] = True
        stored[0]["no_agent"] = no_agent
        stored[0]["script"] = script
        jobs.save_jobs(stored)

    before = jobs.get_job(job["id"])["next_run_at"]
    assert jobs.requeue_interrupted_jobs([
        {"id": "exec-invalid", "job_id": job["id"]}
    ]) == set()
    recovered = jobs.get_job(job["id"])
    assert recovered["next_run_at"] == before
    assert "interrupted_retry" not in recovered


def test_removing_pending_retry_consumes_its_ack(isolated_cron, monkeypatch):
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is True

    # resolve_job_ref may race with recovery and return a pre-marker snapshot;
    # remove_job must capture the current marker only after taking _jobs_lock.
    monkeypatch.setattr(jobs, "resolve_job_ref", lambda _job_id: job)
    assert jobs.remove_job(job["id"]) is True

    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is False


def test_recovery_prunes_ack_created_after_concurrent_policy_cancel(
    isolated_cron, monkeypatch
):
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    job = _script_job(retry_interrupted=True)
    execution = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(execution["id"])
    assert jobs.requeue_interrupted_jobs([execution]) == {job["id"]}

    # Cancellation wins the jobs.json race before terminalization creates its
    # SQLite acknowledgement, so eager cancellation cleanup cannot see it.
    jobs.update_job(job["id"], {"retry_interrupted": False})
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={job["id"]}
    )
    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is True

    assert executions.recover_interrupted_executions() == 0

    assert executions.execution_ids_are_terminal(
        [execution["id"]], job_id=job["id"]
    ) is False


def test_direct_retry_claim_creates_recoverable_attempt_before_consuming_marker(
    isolated_cron, monkeypatch
):
    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )

    claimed = claim_fire_with_execution(job["id"], source="direct")

    assert isinstance(claimed, dict)
    replacement_id = claimed["execution_id"]
    assert executions.latest_execution(job["id"])["id"] == replacement_id
    assert "interrupted_retry" not in jobs.get_job(job["id"])

    # A crash in the old gap (after marker consumption, before run_one_job)
    # now leaves this replacement row for the next recovery pass to requeue.
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started_at: False)
    assert executions.recover_interrupted_executions() == 1
    retry = jobs.get_job(job["id"])["interrupted_retry"]
    assert retry["execution_ids"] == [replacement_id]


@pytest.mark.parametrize(
    "pre_run_failure",
    ["heartbeat_false", "heartbeat_error", "thread_start_error"],
)
def test_pre_run_failure_restores_interrupted_retry(
    isolated_cron, monkeypatch, pre_run_failure
):
    import cron.scheduler as scheduler
    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    before = jobs.get_job(job["id"])
    claimed = claim_fire_with_execution(job["id"], source="direct")

    assert isinstance(claimed, dict)
    assert isinstance(claimed.get("_fire_claim_rollback"), dict)
    replacement_id = claimed["execution_id"]

    if pre_run_failure == "heartbeat_false":
        monkeypatch.setattr(
            scheduler, "heartbeat_fire_claim", lambda *_args, **_kwargs: False
        )
    elif pre_run_failure == "heartbeat_error":
        def _heartbeat_error(*_args, **_kwargs):
            raise OSError("jobs store unavailable")

        monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _heartbeat_error)
    else:
        monkeypatch.setattr(
            scheduler, "heartbeat_fire_claim", lambda *_args, **_kwargs: True
        )

        def _thread_start_error(_thread):
            raise RuntimeError("heartbeat thread unavailable")

        monkeypatch.setattr(scheduler.threading.Thread, "start", _thread_start_error)

    assert scheduler.run_one_job(claimed) is False

    restored = jobs.get_job(job["id"])
    assert restored["fire_claim"] is None
    assert restored["next_run_at"] == before["next_run_at"]
    assert restored["interrupted_retry"] == before["interrupted_retry"]
    assert executions.execution_ids_are_terminal(
        [abandoned["id"]], job_id=job["id"]
    ) is True
    replacement = next(
        row
        for row in executions.list_executions(job_id=job["id"])
        if row["id"] == replacement_id
    )
    assert replacement["status"] == "failed"


def test_pre_run_rollback_io_failure_releases_attempt_for_periodic_recovery(
    isolated_cron, monkeypatch
):
    import cron.scheduler as scheduler
    import cron.jobs as cron_jobs
    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    claimed = claim_fire_with_execution(job["id"], source="direct")
    assert isinstance(claimed, dict)
    replacement_id = claimed["execution_id"]

    def _heartbeat_error(*_args, **_kwargs):
        raise OSError("jobs store unavailable")

    def _rollback_error(_claimed, **_kwargs):
        raise OSError("jobs store still unavailable")

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _heartbeat_error)
    monkeypatch.setattr(cron_jobs, "rollback_fire_claim_setup", _rollback_error)

    assert scheduler.run_one_job(claimed) is False

    replacement = next(
        row
        for row in executions.list_executions(job_id=job["id"])
        if row["id"] == replacement_id
    )
    assert replacement["status"] == "claimed"
    assert replacement["pid"] == -1

    # Once jobs-store I/O recovers, the ordinary periodic recovery path sees
    # the explicitly released owner and reconstructs the owed retry.
    assert executions.recover_interrupted_executions() == 1
    retry = jobs.get_job(job["id"])["interrupted_retry"]
    assert retry["execution_ids"] == [replacement_id]


def test_pre_run_rollback_lock_failure_releases_attempt_for_periodic_recovery(
    isolated_cron, monkeypatch
):
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    claimed = claim_fire_with_execution(job["id"], source="direct")
    assert isinstance(claimed, dict)
    replacement_id = claimed["execution_id"]

    def _heartbeat_error(*_args, **_kwargs):
        raise OSError("jobs store unavailable")

    real_fire_job_lock = cron_jobs._fire_job_lock

    @contextlib.contextmanager
    def _unavailable_fire_job_lock(_job_id):
        yield False

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _heartbeat_error)
    monkeypatch.setattr(cron_jobs, "_fire_job_lock", _unavailable_fire_job_lock)

    assert scheduler.run_one_job(claimed) is False

    replacement = next(
        row
        for row in executions.list_executions(job_id=job["id"])
        if row["id"] == replacement_id
    )
    assert replacement["status"] == "claimed"
    assert replacement["pid"] == -1

    # Recovery also fails closed on the same side-effect fence. Once the lock
    # backend returns, the periodic pass can safely reconstruct the retry.
    monkeypatch.setattr(cron_jobs, "_fire_job_lock", real_fire_job_lock)
    assert executions.recover_interrupted_executions() == 1
    retry = jobs.get_job(job["id"])["interrupted_retry"]
    assert retry["execution_ids"] == [replacement_id]


def test_running_ledger_failure_restores_retry_before_dispatch(
    isolated_cron, monkeypatch
):
    import cron.scheduler as scheduler
    from cron.scheduler_provider import claim_fire_with_execution

    job = _script_job(retry_interrupted=True)
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    before = jobs.get_job(job["id"])
    claimed = claim_fire_with_execution(job["id"], source="direct")
    dispatch_calls = []

    monkeypatch.setattr(
        scheduler,
        "mark_execution_running",
        lambda _execution_id: (_ for _ in ()).throw(OSError("ledger offline")),
    )
    monkeypatch.setattr(
        scheduler,
        "claim_dispatch",
        lambda _job_id: dispatch_calls.append(True) or True,
    )

    assert scheduler.run_one_job(claimed) is False

    restored = jobs.get_job(job["id"])
    assert restored["interrupted_retry"] == before["interrupted_retry"]
    assert restored["next_run_at"] == before["next_run_at"]
    assert restored["fire_claim"] is None
    assert dispatch_calls == []


def test_ambiguous_oneshot_dispatch_is_recovered_with_its_capacity(
    isolated_cron, monkeypatch
):
    import cron.scheduler as scheduler
    from cron.scheduler_provider import claim_fire_with_execution

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
    assert jobs.claim_dispatch(job["id"]) is True
    abandoned = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(abandoned["id"])
    assert jobs.requeue_interrupted_jobs([abandoned]) == {job["id"]}
    executions.mark_interrupted_executions_unknown(
        [abandoned["id"]], retry_job_ids={job["id"]}
    )
    before = jobs.get_job(job["id"])
    assert before["repeat"]["completed"] == 0
    claimed = claim_fire_with_execution(job["id"], source="direct")
    replacement_id = claimed["execution_id"]
    real_claim_dispatch = jobs.claim_dispatch

    def _ambiguous_dispatch(job_id):
        assert real_claim_dispatch(job_id) is True
        raise OSError("save acknowledgement lost")

    monkeypatch.setattr(scheduler, "claim_dispatch", _ambiguous_dispatch)

    assert scheduler.run_one_job(claimed) is False
    recovered = jobs.get_job(job["id"])
    assert recovered["repeat"]["completed"] == 0
    assert recovered["interrupted_retry"] == before["interrupted_retry"]
    assert recovered["next_run_at"] == before["next_run_at"]
    assert recovered["fire_claim"] is None
    assert executions.latest_execution(job["id"])["id"] == replacement_id
    assert executions.latest_execution(job["id"])["status"] == "failed"
    assert executions.recover_interrupted_executions() == 0
