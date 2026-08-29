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

    monkeypatch.setattr(jobs, "_hermes_now", lambda: eligible_at + timedelta(seconds=1))
    assert jobs.get_due_jobs() == []
    executions.mark_interrupted_executions_unknown(
        [execution["id"]], retry_job_ids={opted_in["id"]}
    )
    due = jobs.get_due_jobs()
    assert [job["id"] for job in due] == [opted_in["id"]]
    jobs.advance_next_runs([opted_in["id"]])
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
