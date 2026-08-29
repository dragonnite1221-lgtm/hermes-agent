"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import os
import logging
import socket
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Collection, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# Optional test override. Production resolves the path at transaction time so
# dashboard operations that temporarily enter another profile cannot leak that
# profile's execution records into the import-time home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
_TERMINAL_STATES = ("completed", "failed", "unknown")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


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
    if "machine_id" not in columns:
        conn.execute("ALTER TABLE executions ADD COLUMN machine_id TEXT")
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


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    return current is not None and current == started_at


def _machine_id() -> str:
    """Return the stable host identity used to scope PID liveness checks.

    Shared ledgers spanning hosts must set a unique ``HERMES_MACHINE_ID`` on
    every replica. A hostname fallback preserves safe single-host behavior;
    unlike the process UUID and PID, it remains stable across restarts.
    """
    explicit = os.getenv("HERMES_MACHINE_ID", "").strip()
    return explicit or socket.gethostname().strip()


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


def create_execution(job_id: str, *, source: str) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, machine_id, pid, process_started_at,
                status, claimed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'claimed', ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, _machine_id(), pid,
             _process_start_time(pid), now),
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
            """UPDATE executions SET status='running', started_at=?
               WHERE id=? AND status='claimed'""",
            (now, execution_id),
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
    return record


def discard_unacquired_execution(execution_id: str) -> bool:
    """Remove a ledger-first row when the corresponding job claim lost."""
    with _transaction() as conn:
        cur = conn.execute(
            "DELETE FROM executions WHERE id=? AND status='claimed'",
            (str(execution_id),),
        )
    return cur.rowcount == 1


def interrupted_execution_candidates() -> List[Dict[str, Any]]:
    """Return active attempts whose exact owner process is provably gone."""
    abandoned: List[Dict[str, Any]] = []
    legacy_rows = 0
    local_machine_id = _machine_id()
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT * FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            owner_machine_id = str(row["machine_id"] or "").strip()
            if not owner_machine_id:
                # Legacy rows cannot be assigned to this PID namespace with
                # proof. Fail closed instead of rewriting an active attempt.
                legacy_rows += 1
                continue
            if owner_machine_id != local_machine_id:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            record = _record(row)
            if record is not None:
                abandoned.append(record)
    if legacy_rows:
        logger.warning(
            "Skipped %d active legacy cron execution(s) without machine identity; "
            "ownership cannot be proved",
            legacy_rows,
        )
    return abandoned


def execution_ids_are_terminal(
    execution_ids: Collection[str], *, job_id: str
) -> bool:
    """Return true only when every requested execution exists and is terminal.

    Interrupted-job retry markers are the durable half of a cross-store
    handoff: jobs.json records that a retry is owed, while this SQLite ledger
    records that the abandoned owner can no longer finish.  The retry must not
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
            f"""SELECT id FROM executions
                WHERE id IN ({placeholders})
                  AND job_id = ?
                  AND status IN ('completed','failed','unknown')
                UNION
                SELECT execution_id AS id FROM interrupted_retry_acks
                WHERE execution_id IN ({placeholders})
                  AND job_id = ?""",
            params + (str(job_id),) + params + (str(job_id),),
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


def prune_orphaned_interrupted_retry_acks() -> int:
    """Delete acknowledgements no longer referenced by jobs.json.

    Normal retry claim, policy cancellation, and job removal eagerly consume
    their acknowledgements. This reconciliation closes the cross-store crash
    and race windows: if either process dies between the jobs.json commit and
    SQLite cleanup, the next recovery pass converges the ledger to the actual
    durable retry markers instead of leaking rows forever.
    """
    from cron.jobs import _jobs_lock, load_jobs

    with _jobs_lock():
        active = {
            (str(execution_id), str(job.get("id") or ""))
            for job in load_jobs()
            if isinstance(job, dict)
            for marker in [job.get("interrupted_retry")]
            if isinstance(marker, dict)
            for execution_id in marker.get("execution_ids") or []
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
            cur = conn.execute(
                """UPDATE executions SET status='unknown', finished_at=?, error=?
                   WHERE id=? AND status IN ('claimed','running')""",
                (now, detail, row["id"]),
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


def recover_interrupted_executions() -> int:
    """Classify abandoned attempts after durably applying explicit retry policy."""
    candidates = interrupted_execution_candidates()
    recovered: List[Dict[str, Any]] = []
    if candidates:
        from cron.jobs import requeue_interrupted_jobs

        requeued = requeue_interrupted_jobs(candidates)
        recovered = mark_interrupted_executions_unknown(
            [row["id"] for row in candidates],
            retry_job_ids=requeued,
        )
        if requeued:
            logging.getLogger("cron.scheduler_provider").warning(
                "Durably requeued %d interrupted cron job(s) under explicit at-least-once policy",
                len(requeued),
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
