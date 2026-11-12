"""Job persistence (SQLite) and background workers.

Three independent worker threads:

* extraction      -- original subset extraction jobs (``jobs`` table).
* parameter       -- training-period statistics recomputation (one at a time,
                     enforced at creation and by the single thread).
* normalization   -- apply the parameter snapshot captured at submission time.

Restart policy: every non-terminal job of every kind is marked FAILED, and
partial temp files are removed.  The *active* parameter set is never modified
by a failed or interrupted recomputation; completed parameter jobs and
standardized results stay downloadable.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .engine import ExtractionError, Selection, run_extraction
from .standardize import (StandardizationError, check_target_grid,
                          compute_parameters, load_parameters,
                          run_normalization)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    params      TEXT NOT NULL,
    files       TEXT NOT NULL,
    error       TEXT,
    progress    INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    result      TEXT
);

CREATE TABLE IF NOT EXISTS param_jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    training_jobs TEXT NOT NULL,
    files         TEXT NOT NULL,
    error         TEXT,
    progress      INTEGER NOT NULL DEFAULT 0,
    total         INTEGER NOT NULL DEFAULT 0,
    result        TEXT
);

CREATE TABLE IF NOT EXISTS norm_jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    target_job  TEXT NOT NULL,
    param_job   TEXT,
    param_path  TEXT NOT NULL,
    snapshot    TEXT NOT NULL,
    error       TEXT,
    progress    INTEGER NOT NULL DEFAULT 0,
    total       INTEGER NOT NULL DEFAULT 0,
    result      TEXT
);

CREATE TABLE IF NOT EXISTS active_parameters (
    id          TEXT PRIMARY KEY,
    param_job   TEXT NOT NULL,
    path        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

ACTIVE_PARAM_ID = "current"

_lock = threading.Lock()
# Re-entrant: start_worker holds it while calling recover_on_startup(),
# which may itself initialise the connection via get_db().
_startup_lock = threading.RLock()
_conn: Optional[sqlite3.Connection] = None
_worker_started = False
_param_worker_started = False
_norm_worker_started = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _startup_lock:
            if _conn is None:
                config.DATA_HOME.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(config.DB_PATH),
                                       check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SCHEMA)
                conn.commit()
                _conn = conn
    return _conn


# ---------------------------------------------------------------------------
# Restart recovery
# ---------------------------------------------------------------------------

def recover_on_startup() -> None:
    """Mark every non-terminal job (extraction, parameter, normalization) as
    FAILED and drop partial artifacts.  Active parameters and finished results
    are left untouched."""
    db = get_db()
    interrupted_norm = []
    with _lock:
        rows = db.execute(
            "SELECT id, result FROM jobs "
            "WHERE status IN ('QUEUED','RUNNING')").fetchall()
        for r in rows:
            db.execute(
                "UPDATE jobs SET status='FAILED', updated_at=?, "
                "error=? WHERE id=?",
                (_now(), "服务重启导致任务中断，已标记为失败", r["id"]))
            if r["result"]:
                p = Path(r["result"])
                if p.exists() and p.parent == config.OUTPUT_DIR:
                    try:
                        p.unlink()
                    except OSError:
                        pass

        param_rows = db.execute(
            "SELECT id FROM param_jobs WHERE status IN ('QUEUED','RUNNING')"
        ).fetchall()
        for r in param_rows:
            db.execute(
                "UPDATE param_jobs SET status='FAILED', updated_at=?, "
                "error=? WHERE id=?",
                (_now(), "服务重启导致参数重算中断，已标记为失败；"
                         "原参数保持不变", r["id"]))
            stale = config.PARAM_DIR / f"param_{r['id']}.nc"
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass

        norm_rows = db.execute(
            "SELECT id FROM norm_jobs WHERE status IN ('QUEUED','RUNNING')"
        ).fetchall()
        for r in norm_rows:
            interrupted_norm.append(r["id"])
            db.execute(
                "UPDATE norm_jobs SET status='FAILED', updated_at=?, "
                "error=? WHERE id=?",
                (_now(), "服务重启导致转换中断，已标记为失败", r["id"]))
        db.commit()

    # A crash between atomic rename and status update can leave a complete
    # file for a job that is marked FAILED.  It must never be downloadable.
    for job_id in interrupted_norm:
        stale = config.NORM_OUTPUT_DIR / f"normalized_{job_id}.nc"
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass
    for partial in config.TMP_DIR.glob(".partial-*"):
        try:
            partial.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Extraction jobs (original entry point)
# ---------------------------------------------------------------------------

def create_job(params: dict, stored_files: list) -> str:
    job_id = uuid.uuid4().hex
    db = get_db()
    with _lock:
        db.execute(
            "INSERT INTO jobs (id, status, created_at, updated_at, params, "
            "files, error, progress, total) VALUES (?,?,?,?,?,?,?,0,0)",
            (job_id, "QUEUED", _now(), _now(),
             json.dumps(params, ensure_ascii=False),
             json.dumps(stored_files, ensure_ascii=False), None))
        db.commit()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    db = get_db()
    with _lock:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list:
    db = get_db()
    with _lock:
        rows = db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def list_succeeded_jobs(limit: int = 100) -> list:
    db = get_db()
    with _lock:
        rows = db.execute(
            "SELECT * FROM jobs WHERE status='SUCCEEDED' "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def _update(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [job_id]
    db = get_db()
    with _lock:
        db.execute(f"UPDATE jobs SET {sets} WHERE id=?", values)
        db.commit()


def _claim(table: str, job_id: str) -> bool:
    """Atomically move one QUEUED row to RUNNING.

    Several worker threads may exist (a restart recovery adds no duplicates,
    but tests and repeated ``start_worker`` calls can); exactly one thread
    wins the claim and no job is ever processed twice.
    """
    db = get_db()
    with _lock:
        cur = db.execute(
            f"UPDATE {table} SET status='RUNNING', updated_at=? "
            "WHERE id=? AND status='QUEUED'", (_now(), job_id))
        db.commit()
        return cur.rowcount == 1


def _process_job(row: dict) -> None:
    job_id = row["id"]
    params = json.loads(row["params"])
    files = json.loads(row["files"])
    try:
        sel = Selection(
            t_start=int(params["t_start"]),
            t_end=int(params["t_end"]),
            south=float(params["south"]),
            north=float(params["north"]),
            west=float(params["west"]),
            east=float(params["east"]),
        )
        output_path = str(config.OUTPUT_DIR / f"subset_{job_id}.nc")

        def progress(done: int, total: int) -> None:
            _update(job_id, progress=done, total=total)

        stats = run_extraction(files, sel, output_path, progress)
        # The file is fully written and renamed only at this point, so the
        # SUCCEEDED status and a complete file always appear together.
        _update(job_id, status="SUCCEEDED", progress=stats["records"],
                total=stats["records"], result=output_path,
                error=None)
    except ExtractionError as exc:
        _update(job_id, status="FAILED", error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        _update(job_id, status="FAILED",
                error=f"内部错误: {exc}\n{traceback.format_exc()}")


def _worker_loop() -> None:
    db = get_db()
    while True:
        with _lock:
            row = db.execute(
                "SELECT * FROM jobs WHERE status='QUEUED' "
                "ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            time.sleep(1.0)
            continue
        if not _claim("jobs", row["id"]):
            continue
        _process_job(dict(row))


# ---------------------------------------------------------------------------
# Parameter (statistics) jobs
# ---------------------------------------------------------------------------

def _param_get(job_id: str) -> Optional[dict]:
    db = get_db()
    with _lock:
        row = db.execute("SELECT * FROM param_jobs WHERE id=?",
                         (job_id,)).fetchone()
    return dict(row) if row else None


def list_param_jobs(limit: int = 50) -> list:
    db = get_db()
    with _lock:
        rows = db.execute(
            "SELECT * FROM param_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_active_parameters() -> Optional[dict]:
    """Current effective parameter set, or None."""
    db = get_db()
    with _lock:
        row = db.execute(
            "SELECT * FROM active_parameters WHERE id=?",
            (ACTIVE_PARAM_ID,)).fetchone()
    if row is None:
        return None
    p = Path(row["path"])
    if not p.is_file():
        return None
    return dict(row)


def parameter_recompute_active() -> bool:
    db = get_db()
    with _lock:
        row = db.execute(
            "SELECT id FROM param_jobs WHERE status IN ('QUEUED','RUNNING') "
            "LIMIT 1").fetchone()
    return row is not None


def create_param_job(training_job_ids: list) -> str:
    """Validate the chosen training jobs and enqueue one recomputation.

    Only one recomputation may be pending/running at a time.  The current
    parameter set is not touched here; it is switched atomically only after a
    successful recomputation.
    """
    db = get_db()
    with _lock:
        busy = db.execute(
            "SELECT id FROM param_jobs WHERE status IN ('QUEUED','RUNNING') "
            "LIMIT 1").fetchone()
        if busy is not None:
            raise StandardizationError(
                "已有参数重算正在排队或运行，同一时间只能运行一次参数重算")

        files = []
        for jid in training_job_ids:
            row = db.execute("SELECT * FROM jobs WHERE id=?",
                             (jid,)).fetchone()
            if row is None:
                raise StandardizationError(f"训练作业 {jid} 不存在")
            if row["status"] != "SUCCEEDED" or not row["result"]:
                raise StandardizationError(
                    f"训练作业 {jid} 不是已成功的提取作业")
            p = Path(row["result"])
            if p.parent != config.OUTPUT_DIR or not p.is_file():
                raise StandardizationError(
                    f"训练作业 {jid} 的结果文件不存在")
            files.append(str(p))
        if not files:
            raise StandardizationError("请至少选择一个已成功的提取作业作为训练数据")

        job_id = uuid.uuid4().hex
        db.execute(
            "INSERT INTO param_jobs (id, status, created_at, updated_at, "
            "training_jobs, files, error, progress, total) "
            "VALUES (?,?,?,?,?,?,?,0,0)",
            (job_id, "QUEUED", _now(), _now(),
             json.dumps(list(training_job_ids), ensure_ascii=False),
             json.dumps(files, ensure_ascii=False), None))
        db.commit()
    return job_id


def _param_update(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [job_id]
    db = get_db()
    with _lock:
        db.execute(f"UPDATE param_jobs SET {sets} WHERE id=?", values)
        db.commit()


def _process_param_job(row: dict) -> None:
    job_id = row["id"]
    files = json.loads(row["files"])
    try:
        output_path = str(config.PARAM_DIR / f"param_{job_id}.nc")

        def progress(done: int, total: int) -> None:
            _param_update(job_id, progress=done, total=total)

        stats = compute_parameters(files, output_path, progress)
        # Only a successful recomputation becomes effective.  The job row and
        # the active pointer switch together in one transaction; conversions
        # already in flight hold their own immutable snapshot.
        db = get_db()
        with _lock:
            db.execute(
                "UPDATE param_jobs SET status='SUCCEEDED', updated_at=?, "
                "error=NULL, progress=?, total=?, result=? WHERE id=?",
                (_now(), stats["time_records"], stats["time_records"],
                 output_path, job_id))
            db.execute("DELETE FROM active_parameters")
            db.execute(
                "INSERT INTO active_parameters (id, param_job, path, "
                "created_at) VALUES (?,?,?,?)",
                (ACTIVE_PARAM_ID, job_id, output_path, _now()))
            db.commit()
    except (StandardizationError, ExtractionError) as exc:
        _param_update(job_id, status="FAILED", error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        _param_update(job_id, status="FAILED",
                      error=f"内部错误: {exc}\n{traceback.format_exc()}")


def _param_worker_loop() -> None:
    db = get_db()
    while True:
        with _lock:
            row = db.execute(
                "SELECT * FROM param_jobs WHERE status='QUEUED' "
                "ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            time.sleep(1.0)
            continue
        if not _claim("param_jobs", row["id"]):
            continue
        _process_param_job(dict(row))


# ---------------------------------------------------------------------------
# Normalization jobs
# ---------------------------------------------------------------------------

def _norm_get(job_id: str) -> Optional[dict]:
    db = get_db()
    with _lock:
        row = db.execute("SELECT * FROM norm_jobs WHERE id=?",
                         (job_id,)).fetchone()
    return dict(row) if row else None


def list_norm_jobs(limit: int = 50) -> list:
    db = get_db()
    with _lock:
        rows = db.execute(
            "SELECT * FROM norm_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]


def create_norm_job(target_job_id: str) -> str:
    """Capture the currently effective parameter set and enqueue conversion.

    The parameter file is copied to a per-job snapshot, so a later successful
    recomputation can never change the parameters used by this conversion.
    """
    db = get_db()
    with _lock:
        target = db.execute("SELECT * FROM jobs WHERE id=?",
                            (target_job_id,)).fetchone()
        if target is None:
            raise StandardizationError("目标作业不存在")
        if target["status"] != "SUCCEEDED" or not target["result"]:
            raise StandardizationError("目标作业不是已成功的提取作业")
        tp = Path(target["result"])
        if tp.parent != config.OUTPUT_DIR or not tp.is_file():
            raise StandardizationError("目标作业的结果文件不存在")

        active = db.execute(
            "SELECT * FROM active_parameters WHERE id=?",
            (ACTIVE_PARAM_ID,)).fetchone()
    if active is None:
        raise StandardizationError("当前没有可用的训练参数，请先重算参数")
    param_path = Path(active["path"])
    if not param_path.is_file():
        raise StandardizationError("当前参数文件丢失，请先重算参数")

    # Reject a mismatched target grid before enqueuing.
    try:
        params = load_parameters(str(param_path))
        check_target_grid(str(tp), params)
    except (StandardizationError, ExtractionError) as exc:
        raise StandardizationError(str(exc)) from exc

    job_id = uuid.uuid4().hex
    snapshot_path = config.SNAPSHOT_DIR / f"params_{job_id}.nc"
    try:
        shutil.copy2(str(param_path), str(snapshot_path))
    except OSError as exc:
        raise StandardizationError(f"无法创建参数快照: {exc}") from exc

    db = get_db()
    with _lock:
        db.execute(
            "INSERT INTO norm_jobs (id, status, created_at, updated_at, "
            "target_job, param_job, param_path, snapshot, error, progress, "
            "total) VALUES (?,?,?,?,?,?,?,?,?,0,0)",
            (job_id, "QUEUED", _now(), _now(), target_job_id,
             active["param_job"], str(param_path), str(snapshot_path), None))
        db.commit()
    return job_id


def _norm_update(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [job_id]
    db = get_db()
    with _lock:
        db.execute(f"UPDATE norm_jobs SET {sets} WHERE id=?", values)
        db.commit()


def _process_norm_job(row: dict) -> None:
    job_id = row["id"]
    try:
        target = get_job(row["target_job"])
        if target is None or target["status"] != "SUCCEEDED" \
                or not target["result"]:
            raise StandardizationError("目标提取作业结果不可用")
        target_path = Path(target["result"])
        if target_path.parent != config.OUTPUT_DIR or not target_path.is_file():
            raise StandardizationError("目标提取结果文件不存在")

        # Always use the submission-time snapshot, never the live active set.
        params = load_parameters(row["snapshot"])
        output_path = str(config.NORM_OUTPUT_DIR / f"normalized_{job_id}.nc")

        def progress(done: int, total: int) -> None:
            _norm_update(job_id, progress=done, total=total)

        stats = run_normalization(str(target_path), params, output_path,
                                  progress)
        _norm_update(job_id, status="SUCCEEDED",
                     progress=stats["records"], total=stats["records"],
                     result=output_path, error=None)
    except (StandardizationError, ExtractionError) as exc:
        _norm_update(job_id, status="FAILED", error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        _norm_update(job_id, status="FAILED",
                     error=f"内部错误: {exc}\n{traceback.format_exc()}")


def _norm_worker_loop() -> None:
    db = get_db()
    while True:
        with _lock:
            row = db.execute(
                "SELECT * FROM norm_jobs WHERE status='QUEUED' "
                "ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            time.sleep(1.0)
            continue
        if not _claim("norm_jobs", row["id"]):
            continue
        _process_norm_job(dict(row))


# ---------------------------------------------------------------------------
# Worker startup
# ---------------------------------------------------------------------------

def start_worker() -> None:
    global _worker_started, _param_worker_started, _norm_worker_started
    with _startup_lock:
        if _worker_started and _param_worker_started and _norm_worker_started:
            return
        if not _worker_started:
            # Run recovery exactly once, before any worker thread starts.
            recover_on_startup()
            t = threading.Thread(target=_worker_loop,
                                 name="extraction-worker", daemon=True)
            t.start()
            _worker_started = True
        if not _param_worker_started:
            t = threading.Thread(target=_param_worker_loop,
                                 name="parameter-worker", daemon=True)
            t.start()
            _param_worker_started = True
        if not _norm_worker_started:
            t = threading.Thread(target=_norm_worker_loop,
                                 name="normalization-worker", daemon=True)
            t.start()
            _norm_worker_started = True


# ---------------------------------------------------------------------------
# Download lookups
# ---------------------------------------------------------------------------

def output_path_for(job_id: str) -> Optional[Path]:
    job = get_job(job_id)
    if not job or job["status"] != "SUCCEEDED" or not job["result"]:
        return None
    p = Path(job["result"])
    # Only serve files from the managed output directory.
    if p.parent != config.OUTPUT_DIR or not p.is_file():
        return None
    return p


def normalized_path_for(job_id: str) -> Optional[Path]:
    job = _norm_get(job_id)
    if not job or job["status"] != "SUCCEEDED" or not job["result"]:
        return None
    p = Path(job["result"])
    if p.parent != config.NORM_OUTPUT_DIR or not p.is_file():
        return None
    return p


def param_result_path_for(job_id: str) -> Optional[Path]:
    job = _param_get(job_id)
    if not job or job["status"] != "SUCCEEDED" or not job["result"]:
        return None
    p = Path(job["result"])
    if p.parent != config.PARAM_DIR or not p.is_file():
        return None
    return p


def get_norm_job(job_id: str) -> Optional[dict]:
    return _norm_get(job_id)


def get_param_job(job_id: str) -> Optional[dict]:
    return _param_get(job_id)


def active_param_summary() -> Optional[dict]:
    """Current effective parameters with readable metadata for the UI."""
    active = get_active_parameters()
    if active is None:
        return None
    info = dict(active)
    try:
        params = load_parameters(active["path"])
        count = params["count"]
        std = params["std"]
        if count is None:
            no_training = ~np.isfinite(params["mean"])
            zero_std = np.isfinite(std) & (std == 0.0)
        else:
            no_training = count <= 0
            zero_std = (count > 0) & np.isfinite(std) & (std == 0.0)
        info["lat_count"] = int(params["lat"].size)
        info["lon_count"] = int(params["lon"].size)
        info["valid_cells"] = int(params["valid"].sum())
        info["no_training_cells"] = int(no_training.sum())
        info["zero_std_cells"] = int(zero_std.sum())
        info["missing_cells"] = int((~params["valid"]).sum())
        job = _param_get(active["param_job"]) or {}
        info["training_jobs"] = job.get("training_jobs")
        info["time_records"] = job.get("total")
    except Exception as exc:  # defensive: keep the page usable
        info["error"] = f"无法读取参数文件: {exc}"
    return info
