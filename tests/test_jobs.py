"""Tests for job persistence, restart recovery and the Flask web flow."""
import io
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

# Isolate runtime data before importing the app; share the HOME created by
# the engine test module when both run in one discover process.
_TMP_HOME = Path(os.environ.setdefault(
    "NC_TEST_HOME", tempfile.mkdtemp(prefix="nc-home-job-")))
os.environ["DATA_HOME"] = str(_TMP_HOME)
os.environ["TIME_CHUNK"] = "2"

from app import config, jobs
from app.web import app
from scripts.make_samples import T0, DAY, make_samples


class JobLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nc-jobtests-"))
        cls.info = make_samples(cls.tmp / "samples")
        cls.files = cls.info["files"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(_TMP_HOME, ignore_errors=True)

    def _wait(self, job_id, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = jobs.get_job(job_id)
            if j["status"] in ("SUCCEEDED", "FAILED"):
                return j
            time.sleep(0.1)
        self.fail("job did not finish in time")

    def test_01_success_through_worker(self):
        jobs.start_worker()
        jid = jobs.create_job(
            {"t_start": T0, "t_end": T0 + 3 * DAY,
             "south": -10, "north": 10, "west": 170.0, "east": -170.0},
            self.files)
        job = self._wait(jid)
        self.assertEqual(job["status"], "SUCCEEDED")
        p = jobs.output_path_for(jid)
        self.assertIsNotNone(p)
        self.assertTrue(p.is_file())

    def test_02_conflict_job_failed_no_partial(self):
        jid = jobs.create_job(
            {"t_start": T0, "t_end": T0 + 3 * DAY,
             "south": -10, "north": 10, "west": 170.0, "east": -160.0},
            self.files)
        job = self._wait(jid)
        self.assertEqual(job["status"], "FAILED")
        self.assertIn("冲突", job["error"])
        self.assertIsNone(jobs.output_path_for(jid))
        self.assertEqual(list(config.TMP_DIR.glob(".partial-*")), [])
        self.assertFalse(
            (config.OUTPUT_DIR / f"subset_{jid}.nc").exists())

    def test_03_restart_recovery_marks_unfinished_failed(self):
        jid = jobs.create_job(
            {"t_start": T0, "t_end": T0 + DAY,
             "south": -10, "north": 10, "west": 170.0, "east": -170.0},
            self.files)
        # Simulate a crash before the worker picks the job up: reset the
        # startup hook and run recovery as a fresh process would.
        jobs._worker_started = False
        jobs.recover_on_startup()
        job = jobs.get_job(jid)
        self.assertEqual(job["status"], "FAILED")
        self.assertIn("重启", job["error"])
        # Restarting the worker must not resurrect the recovered job.
        jobs.start_worker()
        time.sleep(1.5)
        self.assertEqual(jobs.get_job(jid)["status"], "FAILED")

    def test_04_web_upload_status_download(self):
        from werkzeug.datastructures import MultiDict
        client = app.test_client()
        form = MultiDict()
        for f in self.files:
            form.add("files",
                     (io.BytesIO(Path(f).read_bytes()), Path(f).name))
        for key, val in [
            ("t_start", str(T0)),
            ("t_end", str(T0 + 3 * DAY)),
            ("south", "-10"), ("north", "10"),
            ("west", "170"), ("east", "-170"),
        ]:
            form.add(key, val)
        resp = client.post("/jobs", data=form,
                           content_type="multipart/form-data",
                           follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        loc = resp.headers["Location"]
        jid = loc.rstrip("/").split("/")[-1]
        job = self._wait(jid)
        self.assertEqual(job["status"], "SUCCEEDED")

        api = client.get(f"/api/jobs/{jid}").get_json()
        self.assertEqual(api["status"], "SUCCEEDED")
        self.assertTrue(api["download"].endswith("/download"))

        dl = client.get(f"/jobs/{jid}/download")
        self.assertEqual(dl.status_code, 200)
        self.assertIn("attachment", dl.headers.get("Content-Disposition", ""))
        magic = dl.data[:4]
        # NetCDF classic/HDF5 magic
        self.assertTrue(magic[:3] == b"CDF" or magic[:4] == b"\x89HDF",
                        f"unexpected file magic: {magic!r}")

        # A failed job must never be downloadable.
        bad = jobs.create_job(
            {"t_start": T0, "t_end": T0 + 3 * DAY,
             "south": -10, "north": 10, "west": 170.0, "east": -160.0},
            self.files)
        self._wait(bad)
        self.assertEqual(client.get(f"/jobs/{bad}/download").status_code, 404)

    def test_05_web_bad_inputs(self):
        client = app.test_client()
        resp = client.post("/jobs", data={
            "t_start": "not-a-time", "t_end": str(T0 + DAY),
            "south": "-10", "north": "10", "west": "170", "east": "-170",
        }, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
