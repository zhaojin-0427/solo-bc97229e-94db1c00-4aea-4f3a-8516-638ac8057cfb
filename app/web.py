"""Flask web entry point: upload daily NetCDF archives, define a subset
selection, poll job status, and download the single merged NetCDF result."""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, url_for)
from werkzeug.utils import secure_filename

from . import config, jobs

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

SAFE_NAME = re.compile(r"^[\w.\-]+$")


def _parse_epoch(value: str, field: str) -> int:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{field}不能为空")
    # Accept either a raw Unix epoch second or an ISO-8601 / datetime-local
    # string (interpreted as UTC).
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    txt = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError as exc:
        raise ValueError(f"{field}时间格式无效: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse_float(value: str, field: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}必须是数字") from exc


@app.before_request
def _ensure_worker() -> None:
    jobs.start_worker()


@app.route("/")
def index():
    return render_template("index.html", jobs=jobs.list_jobs())


@app.route("/jobs", methods=["POST"])
def create_job():
    try:
        t_start = _parse_epoch(request.form.get("t_start", ""), "起始时间")
        t_end = _parse_epoch(request.form.get("t_end", ""), "结束时间")
        south = _parse_float(request.form.get("south", ""), "南界纬度")
        north = _parse_float(request.form.get("north", ""), "北界纬度")
        west = _parse_float(request.form.get("west", ""), "西界经度")
        east = _parse_float(request.form.get("east", ""), "东界经度")
    except ValueError as exc:
        return render_template("error.html", message=str(exc)), 400

    uploads = request.files.getlist("files")
    if not uploads or all(not f.filename for f in uploads):
        return render_template("error.html",
                               message="请至少选择一个 NetCDF 文件"), 400

    job_dir = config.UPLOAD_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    stored = []
    try:
        for seq, f in enumerate(uploads):
            original = f.filename or ""
            ext = os.path.splitext(original)[1].lower()
            if ext not in (".nc", ".netcdf", ".cdf"):
                raise ValueError(f"仅支持 NetCDF 文件: {original}")
            raw_name = secure_filename(original)
            if not raw_name or not SAFE_NAME.fullmatch(raw_name) \
                    or not raw_name.lower().endswith(
                        (".nc", ".netcdf", ".cdf")):
                # Unsafe / non-ASCII / extension-stripped name: generate a
                # safe one while preserving the real extension.
                raw_name = f"upload_{seq}{ext}"
            # Prefix with a sequence to avoid same-name uploads colliding.
            name = f"{seq:03d}_{raw_name}"
            dest = job_dir / name
            f.save(str(dest))
            stored.append(str(dest))
    except Exception as exc:
        for p in stored:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            job_dir.rmdir()
        except OSError:
            pass
        return render_template("error.html", message=str(exc)), 400

    params = {"t_start": t_start, "t_end": t_end, "south": south,
              "north": north, "west": west, "east": east}
    job_id = jobs.create_job(params, stored)
    return redirect(url_for("job_detail", job_id=job_id))


@app.route("/jobs/<job_id>")
def job_detail(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        abort(404)
    return render_template("job.html", job=job)


@app.route("/api/jobs/<job_id>")
def api_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        abort(404)
    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "error": job["error"],
        "download": (url_for("download", job_id=job["id"])
                     if job["status"] == "SUCCEEDED" else None),
    })


@app.route("/jobs/<job_id>/download")
def download(job_id: str):
    p = jobs.output_path_for(job_id)
    if p is None:
        abort(404)
    return send_file(str(p), as_attachment=True,
                     download_name=f"temperature_subset_{job_id}.nc",
                     mimetype="application/x-netcdf")


@app.errorhandler(413)
def too_large(_e):
    return render_template("error.html",
                           message="上传文件超过大小限制"), 413


if __name__ == "__main__":
    jobs.start_worker()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
