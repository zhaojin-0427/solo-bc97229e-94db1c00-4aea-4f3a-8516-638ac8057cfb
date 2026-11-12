"""Gunicorn configuration. A single worker process owns the SQLite job store
and the background extraction thread."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = 1
threads = 8
timeout = 300


def post_worker_init(worker):
    from app import jobs
    # Marks any QUEUED/RUNNING job left by a previous process as FAILED and
    # starts the background extraction thread.
    jobs.start_worker()
