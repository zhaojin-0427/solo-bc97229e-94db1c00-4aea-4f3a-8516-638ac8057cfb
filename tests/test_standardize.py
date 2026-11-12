"""End-to-end tests for training-period statistics and standardization.

Run from the repository root:
    python -m unittest discover -s tests -t . -v
"""
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

# Share the isolated DATA_HOME with the other test modules.
_TMP_HOME = Path(os.environ.setdefault(
    "NC_TEST_HOME", tempfile.mkdtemp(prefix="nc-home-std-")))
os.environ["DATA_HOME"] = str(_TMP_HOME)
os.environ["TIME_CHUNK"] = "2"

import netCDF4 as nc
import numpy as np

from app import config, jobs
from app.engine import Selection, run_extraction
from app.standardize import (StandardizationError, check_target_grid,
                             compute_parameters, load_parameters,
                             run_normalization)
from scripts.make_samples import DAY, T0, make_samples


def _make_grid_file(path, times, values, lat, lon, fill=-9999.0,
                    reverse_lat=False):
    """Write a demo archive with a declarable fill value."""
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[:, ::-1, :] if reverse_lat else vals
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("time", "i8", ("time",))[:] = times
        ds.createVariable("lat", "f8", ("lat",))[:] = (
            lat[::-1] if reverse_lat else lat)
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        v = ds.createVariable("temperature", "f4", ("time", "lat", "lon"),
                              fill_value=fill, compression="zlib", complevel=4)
        v[:] = np.ma.array(vals, mask=np.isnan(vals), fill_value=fill)


class ParameterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = _TMP_HOME
        cls.tmp = Path(tempfile.mkdtemp(prefix="nc-std-"))
        cls._old_chunk = config.TIME_CHUNK
        config.TIME_CHUNK = 2

    @classmethod
    def tearDownClass(cls):
        config.TIME_CHUNK = cls._old_chunk
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _training_set(self):
        """Three training records on a 2x2 grid with one missing cell and
        one constant cell (std = 0) and one all-missing cell."""
        d = self.tmp / "train"
        d.mkdir(exist_ok=True)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 180.0])
        # cell layout values[t, lat, lon]:
        #   t0: [1, 3], [nan, missing]
        #   t1: [3, 3], [nan, missing]
        #   t2: [5, 3], [nan, missing]
        t0, t1, t2 = T0, T0 + 3600, T0 + 7200
        v1 = np.array([
            [[1.0, 3.0], [np.nan, np.nan]],
            [[3.0, 3.0], [np.nan, np.nan]],
            [[5.0, 3.0], [np.nan, np.nan]],
        ], dtype=np.float32)
        # Different time ranges, to exercise the multi-file timestamp merge.
        p1 = d / "part1.nc"
        p2 = d / "part2.nc"
        _make_grid_file(p1, [t0, t1], v1[:2], lat, lon)
        _make_grid_file(p2, [t2], v1[2:], lat, lon, reverse_lat=False)
        return [str(p1), str(p2)], (lat, lon)

    def test_01_statistics_mean_and_population_std_ignore_missing(self):
        files, _ = self._training_set()
        out = str(self.tmp / "params.nc")
        stats = compute_parameters(files, out)
        params = load_parameters(out)
        np.testing.assert_allclose(params["lat"], [0.0, 1.0])
        np.testing.assert_allclose(params["lon"], [0.0, 180.0])
        # cell (lat0, lon0): values 1,3,5 -> mean 3, population std sqrt(8/3)
        self.assertAlmostEqual(float(params["mean"][0, 0]), 3.0, places=6)
        self.assertAlmostEqual(
            float(params["std"][0, 0]), np.sqrt(8.0 / 3.0), places=6)
        # cell (lat0, lon180): constant 3 -> std 0, still "valid" but unusable
        self.assertAlmostEqual(float(params["mean"][0, 1]), 3.0, places=6)
        self.assertAlmostEqual(float(params["std"][0, 1]), 0.0, places=6)
        # cell (lat1, *): all missing -> no valid training value
        self.assertTrue(np.isnan(params["mean"][1, 0]))
        self.assertTrue(np.isnan(params["mean"][1, 1]))
        self.assertFalse(params["valid"][1, 0])
        self.assertFalse(params["valid"][1, 1])
        self.assertEqual(stats["valid_cells"], 2)
        self.assertEqual(stats["zero_std_cells"], 1)
        self.assertEqual(stats["time_records"], 3)
        self.assertEqual(params["count"][0, 0], 3)
        self.assertEqual(params["count"][1, 0], 0)
        # Progress is reported in input-record units across both passes.
        self.assertEqual(list(config.TMP_DIR.glob(".partial-*")), [])

    def test_02_statistics_merge_overlapping_training_records(self):
        d = self.tmp / "overlap"
        d.mkdir(exist_ok=True)
        lat = np.array([0.0])
        lon = np.array([0.0])
        a = d / "a.nc"
        b = d / "b.nc"
        _make_grid_file(a, [T0], np.array([[[2.0]]], dtype=np.float32),
                        lat, lon)
        _make_grid_file(b, [T0, T0 + 3600],
                        np.array([[[2.0]], [[4.0]]], dtype=np.float32),
                        lat, lon)
        out = str(self.tmp / "params_overlap.nc")
        compute_parameters([str(a), str(b)], out)
        params = load_parameters(out)
        self.assertAlmostEqual(float(params["mean"][0, 0]), 3.0, places=6)
        self.assertAlmostEqual(float(params["std"][0, 0]), 1.0, places=6)
        self.assertEqual(params["count"][0, 0], 2)

        # Conflicting duplicate timestamp at the same cell is rejected.
        c = d / "c.nc"
        _make_grid_file(c, [T0], np.array([[[9.0]]], dtype=np.float32),
                        lat, lon)
        with self.assertRaises(Exception):
            compute_parameters([str(a), str(c)], str(self.tmp / "bad.nc"))
        self.assertEqual(list(config.TMP_DIR.glob(".partial-*")), [])

    def test_03_target_grid_mismatch_rejected(self):
        files, grid = self._training_set()
        out = str(self.tmp / "params_grid.nc")
        compute_parameters(files, out)
        params = load_parameters(out)
        other = self.tmp / "other_grid.nc"
        _make_grid_file(other, [T0], np.zeros((1, 2, 2), np.float32),
                        np.array([0.0, 9.0]), np.array([0.0, 180.0]))
        with self.assertRaises(StandardizationError):
            check_target_grid(str(other), params)
        with self.assertRaises(StandardizationError):
            run_normalization(str(other), params,
                              str(self.tmp / "nope.nc"))

    def test_04_normalization_formula_and_missing_cells(self):
        files, grid = self._training_set()
        param_path = str(self.tmp / "params_norm.nc")
        compute_parameters(files, param_path)
        params = load_parameters(param_path)

        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 180.0])
        target = self.tmp / "target.nc"
        # target time records: two; cell values with one missing.
        vals = np.array([
            [[1.0, 3.0], [np.nan, 2.0]],
            [[5.0, 3.0], [np.nan, 4.0]],
        ], dtype=np.float32)
        _make_grid_file(target, [T0 + 10 * DAY, T0 + 10 * DAY + 3600],
                        vals, lat, lon)
        out = str(self.tmp / "norm.nc")
        stats = run_normalization(str(target), params, out)
        self.assertEqual(stats["records"], 2)
        with nc.Dataset(out) as ds:
            # Coordinates preserved exactly.
            np.testing.assert_array_equal(ds.variables["lat"][:], lat)
            np.testing.assert_array_equal(ds.variables["lon"][:], lon)
            np.testing.assert_array_equal(
                ds.variables["time"][:],
                [T0 + 10 * DAY, T0 + 10 * DAY + 3600])
            t = ds.variables["temperature"]
            self.assertTrue(np.issubdtype(t.dtype, np.floating))
            data = t[:]
            mask = np.ma.getmaskarray(data)
            # (lat0, lon0): mean 3, std sqrt(8/3)
            mean = 3.0
            std = np.sqrt(8.0 / 3.0)
            self.assertFalse(mask[0, 0, 0])
            self.assertAlmostEqual(float(data[0, 0, 0]),
                                   (1.0 - mean) / std, places=6)
            self.assertAlmostEqual(float(data[1, 0, 0]),
                                   (5.0 - mean) / std, places=6)
            # (lat0, lon180): std == 0 -> missing in output
            self.assertTrue(mask[0, 0, 1])
            self.assertTrue(mask[1, 0, 1])
            # (lat1, lon0): target missing -> missing
            self.assertTrue(mask[0, 1, 0])
            # (lat1, lon180): target present but no valid training value -> missing
            self.assertTrue(mask[0, 1, 1])
            self.assertTrue(mask[1, 1, 1])
        self.assertEqual(list(config.TMP_DIR.glob(".partial-*")), [])


class StandardizationJobTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nc-stdjob-"))
        cls.info = make_samples(cls.tmp / "samples")
        config.TIME_CHUNK = 2

    @classmethod
    def tearDownClass(cls):
        # Last module in a discover run: clean up the shared DATA_HOME.
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(_TMP_HOME, ignore_errors=True)

    def _wait(self, getter, job_id, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = getter(job_id)
            if j["status"] in ("SUCCEEDED", "FAILED"):
                return j
            time.sleep(0.1)
        self.fail("job did not finish in time")

    def _extract(self, files, t_end, west=170.0, east=-170.0,
                 south=-10.0, north=10.0):
        jid = jobs.create_job(
            {"t_start": T0, "t_end": t_end, "south": south, "north": north,
             "west": west, "east": east}, files)
        job = self._wait(jobs.get_job, jid)
        self.assertEqual(job["status"], "SUCCEEDED")
        return jid

    def _train(self, job_ids):
        pid = jobs.create_param_job(job_ids)
        job = self._wait(jobs.get_param_job, pid)
        self.assertEqual(job["status"], "SUCCEEDED")
        return pid

    def test_01_full_flow_and_parameter_snapshot_isolation(self):
        jobs.start_worker()
        a = self._extract(self.info["files"], T0 + 3 * DAY)
        pid1 = self._train([a])
        active = jobs.get_active_parameters()
        self.assertEqual(active["param_job"], pid1)

        # Start a conversion, then recompute parameters.  The conversion must
        # use its submission-time snapshot even if the new parameters become
        # active before it runs.
        nid = jobs.create_norm_job(a)
        pid2 = self._train([a])
        self.assertNotEqual(pid2, pid1)
        self.assertEqual(jobs.get_active_parameters()["param_job"], pid2)
        norm = self._wait(jobs.get_norm_job, nid)
        self.assertEqual(norm["status"], "SUCCEEDED", norm["error"])
        self.assertEqual(norm["param_job"], pid1)
        snapshot = load_parameters(norm["snapshot"])
        np.testing.assert_array_equal(
            snapshot["mean"], load_parameters(
                jobs.get_param_job(pid1)["result"])["mean"])
        p = jobs.normalized_path_for(nid)
        self.assertIsNotNone(p)
        with nc.Dataset(str(p)) as ds:
            self.assertTrue(np.issubdtype(
                ds.variables["temperature"].dtype, np.floating))
            self.assertEqual(ds.dimensions["time"].size, 6)
            lat = ds.variables["lat"][:]
            self.assertEqual(lat.size, 3)
            self.assertTrue(np.all(np.diff(lat) > 0))

    def test_02_only_one_recompute_at_a_time(self):
        a = self._extract(self.info["files"], T0 + 3 * DAY)
        pid = jobs.create_param_job([a])
        self._wait(jobs.get_param_job, pid)
        # After completion a second request is accepted again.
        pid2 = jobs.create_param_job([a])
        self.assertNotEqual(pid, pid2)
        self._wait(jobs.get_param_job, pid2)
        # Concurrent submission is rejected while one is pending/running.
        orig = jobs.compute_parameters
        gate = {"entered": False}

        def slow(files, out, progress=None, chunk_size=None):
            gate["entered"] = True
            time.sleep(0.2)
            return orig(files, out, progress, chunk_size)

        jobs.compute_parameters = slow
        try:
            pid3 = jobs.create_param_job([a])
            with self.assertRaises(StandardizationError):
                jobs.create_param_job([a])
            self._wait(jobs.get_param_job, pid3)
        finally:
            jobs.compute_parameters = orig

    def test_03_training_job_must_be_succeeded(self):
        bad = jobs.create_job(
            {"t_start": T0, "t_end": T0 + 3 * DAY, "south": -10,
             "north": 10, "west": 170.0, "east": -160.0},
            self.info["files"])
        self._wait(jobs.get_job, bad)
        self.assertEqual(jobs.get_job(bad)["status"], "FAILED")
        with self.assertRaises(StandardizationError):
            jobs.create_param_job([bad])
        with self.assertRaises(StandardizationError):
            jobs.create_norm_job(bad)

    def test_04_norm_requires_active_parameters_and_matching_grid(self):
        a = self._extract(self.info["files"], T0 + 3 * DAY)
        # Different subset => different grid => rejected against active params.
        b = self._extract(self.info["files"], T0 + 3 * DAY,
                          west=160.0, east=-170.0)
        pid = self._train([a])
        with self.assertRaises(StandardizationError):
            jobs.create_norm_job(b)

    def test_05_restart_keeps_parameters_and_marks_norm_failed(self):
        a = self._extract(self.info["files"], T0 + DAY)
        pid = self._train([a])
        active_before = jobs.get_active_parameters()
        self.assertEqual(active_before["param_job"], pid)

        nid = jobs.create_norm_job(a)
        # Simulate a crash: the conversion is still QUEUED and a stray
        # complete file exists (as if renamed just before the crash).
        stray = config.NORM_OUTPUT_DIR / f"normalized_{nid}.nc"
        stray.write_bytes(b"incomplete")
        jobs._worker_started = False
        jobs._param_worker_started = False
        jobs._norm_worker_started = False
        jobs.recover_on_startup()

        norm = jobs.get_norm_job(nid)
        self.assertEqual(norm["status"], "FAILED")
        self.assertIn("重启", norm["error"])
        self.assertIsNone(jobs.normalized_path_for(nid))
        self.assertFalse(stray.exists())
        # Effective parameters and finished conversions are preserved.
        self.assertEqual(jobs.get_active_parameters()["param_job"], pid)
        completed = [n for n in jobs.list_norm_jobs()
                     if n["status"] == "SUCCEEDED"]
        for n in completed:
            self.assertIsNotNone(jobs.normalized_path_for(n["id"]))
        jobs.start_worker()

    def test_06_web_standardization_flow(self):
        from werkzeug.datastructures import MultiDict
        a = self._extract(self.info["files"], T0 + 3 * DAY)
        pid = jobs.create_param_job([a])
        self._wait(jobs.get_param_job, pid)

        from app.web import app
        client = app.test_client()
        page = client.get("/standardize")
        self.assertEqual(page.status_code, 200)
        self.assertIn("标准化".encode(), page.data)

        resp = client.post("/normalize", data={"target_job": a})
        self.assertEqual(resp.status_code, 302)
        loc = resp.headers["Location"]
        nid = loc.rstrip("/").split("/")[-1]
        norm = self._wait(jobs.get_norm_job, nid)
        self.assertEqual(norm["status"], "SUCCEEDED")
        api = client.get(f"/api/normalized/{nid}").get_json()
        self.assertEqual(api["status"], "SUCCEEDED")
        self.assertTrue(api["download"].endswith("/download"))
        dl = client.get(f"/normalized/{nid}/download")
        self.assertEqual(dl.status_code, 200)
        self.assertIn("attachment",
                      dl.headers.get("Content-Disposition", ""))
        magic = dl.data[:4]
        self.assertTrue(magic[:3] == b"CDF" or magic[:4] == b"\x89HDF",
                        f"unexpected file magic: {magic!r}")

        # A failed conversion is never downloadable.
        bad_nid = "0" * 32
        self.assertEqual(
            client.get(f"/normalized/{bad_nid}/download").status_code, 404)

        # Recomputation without training data is rejected.
        resp = client.post("/params", data={})
        self.assertEqual(resp.status_code, 400)

    def test_07_chunked_io_bounded(self):
        """Both statistics and conversion read at most TIME_CHUNK time
        records per variable access, regardless of the total time count."""
        import netCDF4 as nc4
        from app import standardize

        n = config.TIME_CHUNK * 10 + 3
        d = self.tmp / "chunkstd"
        d.mkdir(exist_ok=True)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 180.0])
        vals = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2) + 1.0
        p = d / "long_target.nc"
        _make_grid_file(p, np.arange(T0, T0 + n, dtype=np.int64), vals,
                        lat, lon)

        max_time_read = 0
        orig_getitem = nc4.Variable.__getitem__

        def spy_getitem(self, index):
            nonlocal max_time_read
            if self.name == "temperature" and isinstance(index, tuple) \
                    and hasattr(index[0], "__len__"):
                max_time_read = max(max_time_read, len(index[0]))
            return orig_getitem(self, index)

        nc4.Variable.__getitem__ = spy_getitem
        try:
            param_path = str(d / "params.nc")
            compute_parameters([str(p)], param_path)
            params = load_parameters(param_path)
            norm_out = str(d / "norm.nc")
            run_normalization(str(p), params, norm_out)
        finally:
            nc4.Variable.__getitem__ = orig_getitem

        self.assertGreater(max_time_read, 0)
        self.assertLessEqual(max_time_read, config.TIME_CHUNK)
        with nc.Dataset(norm_out) as ds:
            self.assertEqual(ds.dimensions["time"].size, n)


if __name__ == "__main__":
    unittest.main()
