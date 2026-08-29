"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import os
import hashlib
import logging
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Mapping, Optional, Set

from hermes_cli.sqlite_util import add_column_if_missing
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# Optional test override. Production resolves the path at transaction time so
# dashboard operations that temporarily enter another profile cannot leak that
# profile's execution records into the import-time home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
EXECUTION_OWNER_LEASE_SECONDS = 300
_TERMINAL_STATES = ("completed", "failed", "unknown")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex
_MAX_FOREIGN_LEASE_OBSERVATIONS = 4096
_recovery_intent_lock = threading.RLock()
_recovery_intent_ids: Set[str] = set()


def _connect() -> sqlite3.Connection:
    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=5)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    apply_wal_with_fallback(conn, db_label="cron/executions.db")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             machine_id TEXT,
             boot_id TEXT,
             pid_namespace TEXT,
             heartbeat_at TEXT,
             fire_claim_acquired INTEGER NOT NULL DEFAULT 1,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(executions)").fetchall()
    }
    migrations = (
        ("machine_id", "machine_id TEXT"),
        ("boot_id", "boot_id TEXT"),
        ("pid_namespace", "pid_namespace TEXT"),
        ("heartbeat_at", "heartbeat_at TEXT"),
        ("fire_claim_acquired", "fire_claim_acquired INTEGER NOT NULL DEFAULT 1"),
    )
    for column, ddl in migrations:
        if column not in columns:
            # The shared helper catches the duplicate-column race when two
            # replicas both observe the old schema before either ALTER commits.
            add_column_if_missing(conn, "executions", column, ddl)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS interrupted_retry_acks (
             execution_id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             terminal_at TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS foreign_lease_observations (
             execution_id TEXT NOT NULL,
             observer_boot_id TEXT NOT NULL,
             generation TEXT NOT NULL,
             observed_monotonic REAL NOT NULL,
             PRIMARY KEY (execution_id, observer_boot_id)
           )"""
    )
    _prune_foreign_lease_observations_unlocked(conn)


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, always close.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back
    the transaction; it does not close the connection. Relying on that alone
    leaks a connection (and its WAL/SHM file descriptors) on every call,
    since closing then depends on the garbage collector. Schema init runs
    inside the ``try`` too, so a PRAGMA/DDL failure after a successful
    ``connect()`` still closes the connection instead of leaking it.
    """
    with _lock:
        conn = _connect()
        try:
            _initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> Optional[bool]:
    """Return owner liveness, or ``None`` when PID identity is ambiguous."""
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return None
    if started_at is None:
        return True if pid == os.getpid() else None
    current = _process_start_time(pid)
    if current is None:
        return None
    return current == started_at


def _machine_id() -> str:
    """Return the stable host identity used to scope PID liveness checks.

    This is behavioral ownership data, so it must not depend on an undocumented
    environment escape hatch. Prefer the operating system's persistent machine
    identity and hash it before storage; fall back to stable host hardware and
    hostname material on platforms without ``machine-id``.
    """
    materials: list[str] = []
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            materials.append(value)
            break
    if not materials:
        materials.extend((socket.gethostname().strip(), f"{uuid.getnode():012x}"))
    digest = hashlib.sha256("\0".join(materials).encode("utf-8")).hexdigest()
    return f"hermes-host-{digest[:32]}"


def _pid_namespace_id() -> str:
    """Return the PID table identity that makes a numeric PID meaningful."""
    if os.name != "posix":
        return "native-pid-table"
    try:
        namespace = os.readlink("/proc/self/ns/pid")
    except OSError:
        # Without a namespace identity, never compare another process's PID
        # locally. A process-scoped value routes ownership through the safer
        # distributed lease path instead.
        return f"unknown-{_PROCESS_ID}"
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    return f"hermes-pidns-{digest[:32]}"


def _boot_id() -> str:
    """Return the globally random kernel-boot identity for PID attribution."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        value = ""
    if not value:
        # A process-scoped fallback disables cross-process local PID checks and
        # durable monotonic observations on platforms without Linux boot IDs.
        return f"unknown-{_PROCESS_ID}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"hermes-boot-{digest[:32]}"


def _foreign_owner_lease_is_stale(
    row: sqlite3.Row, *, fire_claim_generation: Optional[str] = None
) -> bool:
    """Expire a foreign lease only after one observer sees no renewal locally.

    Persisted heartbeat timestamps are generations, not comparable clocks:
    two replicas can disagree about wall time by more than the lease.  Each
    observer therefore measures an unchanged generation with its own monotonic
    clock. Process restart resets the observation and safely delays recovery
    by one lease instead of risking a duplicate execution.
    """
    raw = row["heartbeat_at"] or row["claimed_at"]
    execution_id = str(row["id"] or "")
    generation = str(raw or "")
    if fire_claim_generation:
        # jobs.json is the authoritative dispatch fence. Include its exact
        # owner/heartbeat generation in the observer-local lease so a runner
        # that can still renew the fire claim is never reaped merely because
        # its best-effort audit-ledger heartbeat is temporarily unavailable.
        generation = f"{generation}\0{fire_claim_generation}"
    if not execution_id or not generation:
        return False
    observed_now = time.monotonic()
    observer_boot_id = _boot_id()
    with _transaction() as conn:
        previous = conn.execute(
            """SELECT generation, observed_monotonic
               FROM foreign_lease_observations
               WHERE execution_id=? AND observer_boot_id=?""",
            (execution_id, observer_boot_id),
        ).fetchone()
        if (
            previous is None
            or str(previous["generation"]) != generation
            or observed_now < float(previous["observed_monotonic"])
        ):
            conn.execute(
                """INSERT INTO foreign_lease_observations
                   (execution_id, observer_boot_id, generation, observed_monotonic)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(execution_id, observer_boot_id) DO UPDATE SET
                     generation=excluded.generation,
                     observed_monotonic=excluded.observed_monotonic""",
                (execution_id, observer_boot_id, generation, observed_now),
            )
            _prune_foreign_lease_observations_unlocked(conn)
            return False
        return (
            observed_now - float(previous["observed_monotonic"])
            >= EXECUTION_OWNER_LEASE_SECONDS
        )


def _owner_lease_is_stale(
    row: sqlite3.Row,
    *,
    local_machine_id: str,
    local_boot_id: str,
    local_pid_namespace: str,
    fire_claim_generation: Optional[str] = None,
) -> bool:
    """Prove an owner dead using PIDs only inside the same PID namespace."""
    owner_machine_id = str(row["machine_id"] or "").strip()
    owner_boot_id = str(row["boot_id"] or "").strip()
    owner_pid_namespace = str(row["pid_namespace"] or "").strip()
    if (
        owner_machine_id == local_machine_id
        and owner_boot_id == local_boot_id
        and owner_pid_namespace == local_pid_namespace
    ):
        owner_is_live = _owner_is_live(
            int(row["pid"]), row["process_started_at"]
        )
        if owner_is_live is not None:
            return not owner_is_live
        # A present PID without a verifiable start-time fingerprint is not
        # proof of either life or death (procfs may be transiently unreadable,
        # and legacy rows may lack the fingerprint). Measure its unchanged
        # durable heartbeat/fire-claim generation with the observer-local
        # lease instead of immediately reaping a potentially live execution.
        return _foreign_owner_lease_is_stale(
            row, fire_claim_generation=fire_claim_generation
        )
    # Missing legacy identities and same-host containers are both foreign PID
    # tables. Their persisted heartbeat generation is the only safe proof.
    return _foreign_owner_lease_is_stale(
        row, fire_claim_generation=fire_claim_generation
    )


def _prune_foreign_lease_observations_unlocked(conn: sqlite3.Connection) -> None:
    """Keep only bounded observations for executions that are still active."""
    conn.execute(
        """DELETE FROM foreign_lease_observations
           WHERE NOT EXISTS (
             SELECT 1 FROM executions
             WHERE executions.id = foreign_lease_observations.execution_id
               AND executions.status IN ('claimed','running')
           )"""
    )
    conn.execute(
        """DELETE FROM foreign_lease_observations WHERE rowid IN (
             SELECT rowid FROM foreign_lease_observations
             ORDER BY rowid DESC LIMIT -1 OFFSET ?
           )""",
        (max(0, int(_MAX_FOREIGN_LEASE_OBSERVATIONS)),),
    )


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )
    _prune_foreign_lease_observations_unlocked(conn)


def create_execution(
    job_id: str, *, source: str, fire_claim_acquired: bool = True
) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, machine_id, boot_id, pid_namespace,
                pid, process_started_at,
                status, claimed_at, heartbeat_at, fire_claim_acquired)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'claimed', ?, ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, _machine_id(),
             _boot_id(), _pid_namespace_id(), pid, _process_start_time(pid), now,
             now, int(fire_claim_acquired)),
        )
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    record = _record(row)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET status='running', started_at=?, heartbeat_at=?
               WHERE id=? AND status='claimed' AND fire_claim_acquired=1""",
            (now, now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET status=?, finished_at=?, error=?
               WHERE id=? AND status IN ('claimed','running')""",
            (status, now, detail, execution_id),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    if record is not None:
        with _recovery_intent_lock:
            _recovery_intent_ids.discard(str(execution_id))
    return record


def release_execution_for_recovery(
    execution_id: str, *, error: Optional[str] = None
) -> bool:
    """Durably relinquish an unstarted attempt to interrupted recovery.

    This is the fallback when the exact jobs-store rollback cannot be written.
    Keeping the acquired execution active preserves the cross-store witness;
    replacing its local owner identity with a provably dead PID makes the next
    periodic recovery pass classify and requeue it without waiting for the
    still-healthy gateway process to restart.
    """
    execution_id = str(execution_id)
    # Record intent before the fallible SQLite write.  If both this update and
    # the caller's terminal fallback fail during an outage, the same live
    # process must still be able to distinguish this deliberately relinquished
    # attempt from its genuinely running executions on the next recovery pass.
    remember_execution_recovery_intent(execution_id)
    now = _hermes_now().isoformat()
    detail = str(error) if error else "Pre-run setup aborted before execution."
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET process_id=?, pid=-1, process_started_at=NULL,
                   heartbeat_at=?, error=?
               WHERE id=? AND status IN ('claimed','running')
                 AND fire_claim_acquired=1""",
            (f"released:{uuid.uuid4().hex}", now, detail, execution_id),
        )
    return cur.rowcount == 1


def remember_execution_recovery_intent(execution_id: str) -> None:
    """Keep a process-local witness when durable relinquish is unavailable.

    The ledger-first fire path can lose both its SQLite promotion and the
    jobs-store rollback acknowledgement during the same outage.  Recording
    this intent before either fallible cleanup lets this still-live process
    reconcile its own provisional row instead of waiting for a restart to
    make the owner lease look abandoned.
    """
    with _recovery_intent_lock:
        _recovery_intent_ids.add(str(execution_id))


def discard_unacquired_execution(execution_id: str) -> bool:
    """Remove a ledger-first row when the corresponding job claim lost."""
    execution_id = str(execution_id)
    with _transaction() as conn:
        cur = conn.execute(
            """DELETE FROM executions
               WHERE id=? AND status='claimed' AND fire_claim_acquired=0""",
            (execution_id,),
        )
    if cur.rowcount == 1:
        with _recovery_intent_lock:
            _recovery_intent_ids.discard(execution_id)
    return cur.rowcount == 1


def mark_fire_claim_acquired(execution_id: str) -> Optional[Dict[str, Any]]:
    """Idempotently fence an attempt after its jobs-store fire CAS succeeds."""
    now = _hermes_now().isoformat()
    transitioned = False
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET fire_claim_acquired=1, heartbeat_at=?
               WHERE id=? AND status='claimed' AND fire_claim_acquired=0""",
            (now, str(execution_id)),
        )
        transitioned = cur.rowcount == 1
        record = _record(
            conn.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
        )
        if (
            record is None
            or record.get("status") != "claimed"
            or int(record.get("fire_claim_acquired") or 0) != 1
        ):
            return None
    if transitioned:
        _emit_execution_state(record)
    return record


def heartbeat_execution(execution_id: str) -> bool:
    """Renew the distributed owner lease for an acquired active attempt."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET heartbeat_at=?
               WHERE id=? AND status IN ('claimed','running')
                 AND fire_claim_acquired=1""",
            (now, str(execution_id)),
        )
    return cur.rowcount == 1


def reconcile_unacquired_executions() -> tuple[int, int]:
    """Promote dead CAS winners and discard dead CAS losers.

    Ledger-first dispatch intentionally persists an unacquired row before the
    jobs-store CAS. The winning CAS records the execution id in ``fire_claim``;
    that durable cross-store witness lets recovery distinguish a winner killed
    before its SQLite promotion from a loser killed before local cleanup.
    """
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT * FROM executions
               WHERE status='claimed' AND fire_claim_acquired=0"""
        ).fetchall()
    if not rows:
        return (0, 0)

    local_machine_id = _machine_id()
    local_boot_id = _boot_id()
    local_pid_namespace = _pid_namespace_id()
    abandoned: list[sqlite3.Row] = []
    for row in rows:
        execution_id = str(row["id"])
        with _recovery_intent_lock:
            locally_released = execution_id in _recovery_intent_ids
        if not locally_released and not _owner_lease_is_stale(
            row,
            local_machine_id=local_machine_id,
            local_boot_id=local_boot_id,
            local_pid_namespace=local_pid_namespace,
        ):
            continue
        abandoned.append(row)
    if not abandoned:
        return (0, 0)

    from cron.jobs import _jobs_lock, fire_recovery_fence, load_jobs

    abandoned_by_job: Dict[str, List[sqlite3.Row]] = {}
    for row in abandoned:
        abandoned_by_job.setdefault(str(row["job_id"]), []).append(row)

    promoted = discarded = 0
    discarded_intents: Set[str] = set()
    for job_id, job_rows in abandoned_by_job.items():
        # The winner lookup and SQLite transition are one per-job fenced
        # operation. A replacement claim therefore cannot land between a
        # stale winners snapshot and promotion of the old execution.
        with fire_recovery_fence(job_id) as fence_acquired:
            if not fence_acquired:
                continue
            with _jobs_lock():
                current_job = next(
                    (
                        job
                        for job in load_jobs()
                        if isinstance(job, dict) and str(job.get("id")) == job_id
                    ),
                    None,
                )
                current_claim = (
                    current_job.get("fire_claim")
                    if isinstance(current_job, dict)
                    else None
                )
                winner_id = (
                    str(current_claim.get("execution_id"))
                    if isinstance(current_claim, dict)
                    and current_claim.get("execution_id")
                    else None
                )
                with _transaction() as conn:
                    for row in job_rows:
                        execution_id = str(row["id"])
                        if execution_id == winner_id:
                            cur = conn.execute(
                                """UPDATE executions SET fire_claim_acquired=1
                                   WHERE id=? AND job_id=? AND status='claimed'
                                     AND fire_claim_acquired=0""",
                                (execution_id, job_id),
                            )
                            promoted += cur.rowcount
                        else:
                            cur = conn.execute(
                                """DELETE FROM executions
                                   WHERE id=? AND job_id=? AND status='claimed'
                                     AND fire_claim_acquired=0""",
                                (execution_id, job_id),
                            )
                            discarded += cur.rowcount
                            if cur.rowcount:
                                discarded_intents.add(execution_id)
    if discarded_intents:
        with _recovery_intent_lock:
            _recovery_intent_ids.difference_update(discarded_intents)
    return promoted, discarded


def _fire_claim_generation_map(
    stored_jobs: Collection[Dict[str, Any]],
) -> Dict[str, str]:
    """Map execution ids to exact authoritative jobs-store lease generations."""
    return {
        str(claim.get("execution_id")): (
            f"{claim.get('by') or ''}:{claim.get('at') or ''}"
        )
        for job in stored_jobs
        if isinstance(job, dict)
        for claim in [job.get("fire_claim")]
        if isinstance(claim, dict) and claim.get("execution_id")
    }


def interrupted_execution_candidates() -> List[Dict[str, Any]]:
    """Return active attempts whose exact owner process is provably gone."""
    from cron.jobs import load_jobs

    try:
        stored_jobs = load_jobs()
    except Exception:
        # The jobs store carries the authoritative fire-claim heartbeat. Fail
        # closed when it cannot be read: an audit-only lease must not authorize
        # a duplicate side effect while dispatch ownership is unknown.
        logger.warning(
            "Skipping interrupted execution recovery because jobs store "
            "ownership could not be verified",
            exc_info=True,
        )
        return []
    fire_claim_generations = _fire_claim_generation_map(stored_jobs)
    abandoned: List[Dict[str, Any]] = []
    local_machine_id = _machine_id()
    local_boot_id = _boot_id()
    local_pid_namespace = _pid_namespace_id()
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT * FROM executions
               WHERE status IN ('claimed','running') AND fire_claim_acquired=1"""
        ).fetchall()
    for row in rows:
        execution_id = str(row["id"])
        with _recovery_intent_lock:
            locally_released = execution_id in _recovery_intent_ids
        if row["process_id"] == _PROCESS_ID and not locally_released:
            continue
        if not locally_released and not _owner_lease_is_stale(
            row,
            local_machine_id=local_machine_id,
            local_boot_id=local_boot_id,
            local_pid_namespace=local_pid_namespace,
            fire_claim_generation=fire_claim_generations.get(str(row["id"])),
        ):
            continue
        record = _record(row)
        if record is not None:
            record["_fire_claim_generation"] = fire_claim_generations.get(execution_id)
            abandoned.append(record)
    return abandoned


def execution_ids_are_terminal(
    execution_ids: Collection[str], *, job_id: str
) -> bool:
    """Return true only when every requested retry handoff has a durable ACK.

    Interrupted-job retry markers are the durable half of a cross-store
    handoff: jobs.json records that a retry is owed, while this SQLite ledger
    records that recovery terminalized that exact abandoned owner. A generic
    completed/failed row is deliberately insufficient because the owner may
    have renewed after the recovery candidate read. The retry must not
    be claimed until the ledger half has committed, otherwise a failure between
    the two writes can execute the same retry again after every restart.
    """
    ids = {str(value) for value in execution_ids if value}
    if not ids:
        return False
    placeholders = ",".join("?" for _ in ids)
    params = tuple(sorted(ids))
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT execution_id AS id FROM interrupted_retry_acks
                WHERE execution_id IN ({placeholders})
                  AND job_id = ?""",
            params + (str(job_id),),
        ).fetchall()
    return {str(row["id"]) for row in rows} == ids


def forget_interrupted_retry_acks(
    execution_ids: Collection[str], *, job_id: str
) -> None:
    """Delete consumed retry acknowledgements after jobs.json commits."""
    ids = {str(value) for value in execution_ids if value}
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with _transaction() as conn:
        conn.execute(
            f"""DELETE FROM interrupted_retry_acks
                WHERE execution_id IN ({placeholders}) AND job_id = ?""",
            tuple(sorted(ids)) + (str(job_id),),
        )


def remember_interrupted_retry_acks(
    execution_ids: Collection[str], *, job_id: str
) -> None:
    """Recreate terminal handoff proof before a jobs-store marker rollback.

    Callers hold the jobs lock, matching the pruner's jobs→SQLite lock order.
    Inserting before restoring the marker makes every crash boundary safe:
    an orphan ACK is prunable, while a restored marker always has its proof.
    """
    ids = {str(value) for value in execution_ids if value}
    if not ids:
        return
    terminal_at = _hermes_now().isoformat()
    with _transaction() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO interrupted_retry_acks
               (execution_id, job_id, terminal_at) VALUES (?, ?, ?)""",
            [(execution_id, str(job_id), terminal_at) for execution_id in sorted(ids)],
        )


def interrupted_retry_states(
    execution_ids: Collection[str],
) -> tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """Return ledger rows and durable retry ACKs for jobs-store reconciliation."""
    ids = {str(value) for value in execution_ids if value}
    if not ids:
        return {}, set()
    placeholders = ",".join("?" for _ in ids)
    params = tuple(sorted(ids))
    with _transaction() as conn:
        rows = conn.execute(
            f"SELECT * FROM executions WHERE id IN ({placeholders})", params
        ).fetchall()
        ack_rows = conn.execute(
            f"""SELECT execution_id FROM interrupted_retry_acks
                WHERE execution_id IN ({placeholders})""",
            params,
        ).fetchall()
    return (
        {str(row["id"]): dict(row) for row in rows},
        {str(row["execution_id"]) for row in ack_rows},
    )


def prune_orphaned_interrupted_retry_acks() -> int:
    """Delete acknowledgements no longer referenced by jobs.json.

    Normal retry claim, policy cancellation, and job removal eagerly consume
    their acknowledgements. This reconciliation closes the cross-store crash
    and race windows: if either process dies between the jobs.json commit and
    SQLite cleanup, the next recovery pass converges the ledger to the actual
    durable retry markers instead of leaking rows forever.
    """
    from cron.jobs import (
        _interrupted_retry_execution_ids,
        _jobs_lock,
        load_jobs,
    )

    with _jobs_lock():
        active = {
            (str(execution_id), str(job.get("id") or ""))
            for job in load_jobs()
            if isinstance(job, dict)
            for marker in [job.get("interrupted_retry")]
            if isinstance(marker, dict)
            for execution_id in _interrupted_retry_execution_ids(marker)
            if execution_id and job.get("id")
        }
        with _transaction() as conn:
            rows = conn.execute(
                "SELECT execution_id, job_id FROM interrupted_retry_acks"
            ).fetchall()
            orphaned = [
                (str(row["execution_id"]), str(row["job_id"]))
                for row in rows
                if (str(row["execution_id"]), str(row["job_id"])) not in active
            ]
            conn.executemany(
                "DELETE FROM interrupted_retry_acks WHERE execution_id=? AND job_id=?",
                orphaned,
            )
    return len(orphaned)


def mark_interrupted_executions_unknown(
    execution_ids: Collection[str], *, retry_job_ids: Collection[str] = (),
    expected_heartbeats: Optional[Mapping[str, Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """CAS abandoned attempts to unknown after any durable retry requeue."""
    ids = {str(value) for value in execution_ids if value}
    retry_ids = {str(value) for value in retry_job_ids if value}
    if not ids:
        return []
    now = _hermes_now().isoformat()
    recovered: List[Dict[str, Any]] = []
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT id, job_id FROM executions WHERE status IN ('claimed','running')"
        ).fetchall()
        for row in rows:
            if row["id"] not in ids:
                continue
            detail = (
                "Scheduler restarted after this execution's owner exited before a durable "
                "terminal state; whether side effects ran is unknown."
            )
            if row["job_id"] in retry_ids:
                detail += " A retry was durably scheduled by the job's explicit interrupted-run policy."
            expected_heartbeat = (
                expected_heartbeats.get(str(row["id"]))
                if expected_heartbeats is not None
                else None
            )
            heartbeat_fence = (
                " AND heartbeat_at IS ?" if expected_heartbeats is not None else ""
            )
            cur = conn.execute(
                f"""UPDATE executions SET status='unknown', finished_at=?, error=?
                    WHERE id=? AND status IN ('claimed','running'){heartbeat_fence}""",
                (now, detail, row["id"], expected_heartbeat)
                if expected_heartbeats is not None
                else (now, detail, row["id"]),
            )
            if cur.rowcount:
                if row["job_id"] in retry_ids:
                    conn.execute(
                        """INSERT OR IGNORE INTO interrupted_retry_acks
                           (execution_id, job_id, terminal_at) VALUES (?, ?, ?)""",
                        (row["id"], row["job_id"], now),
                    )
                record = _record(conn.execute(
                    "SELECT * FROM executions WHERE id=?", (row["id"],)
                ).fetchone())
                if record is not None:
                    recovered.append(record)
        if recovered:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return recovered


def reconcile_pre_run_abort_executions() -> int:
    """Terminalize rollback-restored attempts without creating a retry.

    ``rollback_fire_claim_setup`` restores the owed occurrence and writes an
    execution-id tombstone in the same jobs.json commit. If executions.db was
    unavailable when the abort happened, this fenced pass closes the dangling
    ledger row before generic interrupted recovery can mistake it for work
    whose side effects may have started.
    """
    from cron.jobs import _jobs_lock, fire_recovery_fence, load_jobs, save_jobs

    try:
        snapshot = load_jobs()
    except Exception:
        logger.warning(
            "Skipping pre-run-abort reconciliation because jobs store "
            "could not be read",
            exc_info=True,
        )
        return 0
    job_ids = [
        str(job.get("id"))
        for job in snapshot
        if isinstance(job, dict)
        and isinstance(job.get("_pre_run_abort_execution_ids"), list)
        and job.get("_pre_run_abort_execution_ids")
        and job.get("id")
    ]
    terminalized = 0
    for job_id in job_ids:
        try:
            with fire_recovery_fence(job_id) as fence_acquired:
                if not fence_acquired:
                    continue
                with _jobs_lock():
                    stored_jobs = load_jobs()
                    job = next(
                        (
                            item
                            for item in stored_jobs
                            if isinstance(item, dict)
                            and str(item.get("id")) == job_id
                        ),
                        None,
                    )
                    if not isinstance(job, dict):
                        continue
                    raw_ids = job.get("_pre_run_abort_execution_ids")
                    execution_ids = (
                        {str(value) for value in raw_ids if value}
                        if isinstance(raw_ids, list)
                        else set()
                    )
                    resolved_ids: Set[str] = set()
                    for execution_id in execution_ids:
                        try:
                            # A None result still proves the transaction was
                            # readable and the row was already terminal/missing.
                            finish_execution(
                                execution_id,
                                success=False,
                                error="Pre-run setup aborted before execution.",
                            )
                        except Exception:
                            logger.warning(
                                "Job '%s': pre-run-abort execution %s is still "
                                "not terminalizable",
                                job_id,
                                execution_id,
                                exc_info=True,
                            )
                            continue
                        resolved_ids.add(execution_id)
                    if not resolved_ids:
                        continue
                    remaining = execution_ids - resolved_ids
                    if remaining:
                        job["_pre_run_abort_execution_ids"] = sorted(remaining)
                    else:
                        job.pop("_pre_run_abort_execution_ids", None)
                    save_jobs(stored_jobs)
                    terminalized += len(resolved_ids)
        except Exception:
            logger.warning(
                "Job '%s': pre-run-abort reconciliation did not commit",
                job_id,
                exc_info=True,
            )
    return terminalized


def recover_interrupted_executions() -> int:
    """Classify abandoned attempts after durably applying explicit retry policy."""
    from cron.jobs import reconcile_interrupted_retry_markers

    pre_run_aborts = reconcile_pre_run_abort_executions()
    if pre_run_aborts:
        logger.info(
            "Terminalized %d rollback-restored pre-run abort(s)",
            pre_run_aborts,
        )
    reconcile_interrupted_retry_markers()
    promoted, discarded = reconcile_unacquired_executions()
    if promoted or discarded:
        logger.info(
            "Reconciled abandoned fire-claim setup rows: winners=%d losers=%d",
            promoted,
            discarded,
        )
    candidates = interrupted_execution_candidates()
    recovered: List[Dict[str, Any]] = []
    if candidates:
        from cron.jobs import (
            _jobs_lock,
            cancel_interrupted_retry_executions,
            fire_recovery_fence,
            load_jobs,
            requeue_interrupted_jobs,
        )

        # Serialize each job's recovery with its external side effects before
        # taking the global jobs lock. Heartbeats use only the jobs CAS, so a
        # live owner can keep changing its generation while recovery waits on
        # the fence. Once acquired, revalidate that generation and commit the
        # jobs marker + SQLite terminal CAS as one fenced recovery operation.
        candidates_by_job: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            candidates_by_job.setdefault(str(candidate["job_id"]), []).append(candidate)
        all_requeued: Set[str] = set()
        for job_id, job_candidates in candidates_by_job.items():
            with fire_recovery_fence(job_id) as fence_acquired:
                if not fence_acquired:
                    continue
                with _jobs_lock():
                    current_generations = _fire_claim_generation_map(load_jobs())
                    job_candidates = [
                        row
                        for row in job_candidates
                        if current_generations.get(str(row["id"]))
                        == row.get("_fire_claim_generation")
                    ]
                    if not job_candidates:
                        continue
                    candidate_heartbeats = {
                        str(row["id"]): row.get("heartbeat_at")
                        for row in job_candidates
                    }
                    requeued = requeue_interrupted_jobs(
                        job_candidates, expected_heartbeats=candidate_heartbeats
                    )
                    job_recovered = mark_interrupted_executions_unknown(
                        [row["id"] for row in job_candidates],
                        retry_job_ids=requeued,
                        expected_heartbeats=candidate_heartbeats,
                    )
                    recovered.extend(job_recovered)
                    all_requeued.update(requeued)
                    recovered_ids = {str(row["id"]) for row in job_recovered}
                    lost_ids = {
                        str(row["id"]) for row in job_candidates
                    } - recovered_ids
                    # Another replica may have won the identical terminalization
                    # CAS after this process requeued the marker. Its unknown+ACK
                    # transaction proves the retry handoff succeeded; only an
                    # owner renewal should roll it back.
                    _, concurrently_recovered_ids = interrupted_retry_states(lost_ids)
                    lost_ids -= concurrently_recovered_ids
                    if lost_ids:
                        cancel_interrupted_retry_executions(lost_ids)
        if recovered:
            with _recovery_intent_lock:
                _recovery_intent_ids.difference_update(
                    str(row["id"]) for row in recovered
                )
        if all_requeued:
            logging.getLogger("cron.scheduler_provider").warning(
                "Durably requeued %d interrupted cron job(s) under explicit at-least-once policy",
                len(all_requeued),
            )
    try:
        pruned = prune_orphaned_interrupted_retry_acks()
        if pruned:
            logger.info("Pruned %d orphaned interrupted-retry acknowledgement(s)", pruned)
    except Exception as exc:
        # Recovery classification has already committed. Keep that durable
        # progress and retry this idempotent reconciliation next cycle.
        logger.warning("Interrupted-retry acknowledgement reconciliation failed: %s", exc)
    return len(recovered)


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
