"""Job persistence (SQLite) and the single background worker.

Status lifecycle: QUEUED -> RUNNING -> SUCCEEDED / FAILED.
On application start-up every non-terminal job is marked FAILED, so a restart
can never expose a SUCCEEDED row without a complete, atomically published file.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .engine import ExtractionError, Selection, run_extraction

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
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_worker_started = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DATA_HOME.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


def recover_on_startup() -> None:
    """Mark any job left in a non-terminal state by a previous process as
    FAILED and drop its partial temp/output artifacts."""
    db = get_db()
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
                # A RUNNING job never published via os.replace; remove stale
                # files defensively.
                if p.exists() and p.parent == config.OUTPUT_DIR:
                    try:
                        p.unlink()
                    except OSError:
                        pass
        db.commit()
    for partial in config.TMP_DIR.glob(".partial-*"):
        try:
            partial.unlink()
        except OSError:
            pass


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


def _process_job(row: dict) -> None:
    job_id = row["id"]
    params = json.loads(row["params"])
    files = json.loads(row["files"])
    _update(job_id, status="RUNNING")
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
        _process_job(dict(row))


def start_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    recover_on_startup()
    t = threading.Thread(target=_worker_loop, name="extraction-worker",
                         daemon=True)
    t.start()
    _worker_started = True


def output_path_for(job_id: str) -> Optional[Path]:
    job = get_job(job_id)
    if not job or job["status"] != "SUCCEEDED" or not job["result"]:
        return None
    p = Path(job["result"])
    # Only serve files from the managed output directory.
    if p.parent != config.OUTPUT_DIR or not p.is_file():
        return None
    return p
