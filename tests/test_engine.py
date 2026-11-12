"""End-to-end tests for the streaming extraction engine and job layer.

Run from the repository root:
    python -I -m unittest discover -s tests -v
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Isolate all runtime data before importing the app package. The two test
# modules run in one process under `unittest discover`, so share one HOME.
_TMP_HOME = Path(os.environ.setdefault(
    "NC_TEST_HOME", tempfile.mkdtemp(prefix="nc-home-")))
os.environ["DATA_HOME"] = str(_TMP_HOME)

import netCDF4 as nc
import numpy as np

from app import config
from app.engine import (ExtractionError, Selection, normalize_lon,
                        run_extraction, validate_files)
from scripts.make_samples import T0, DAY, make_samples


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = _TMP_HOME
        cls.tmp = Path(tempfile.mkdtemp(prefix="nc-tests-"))
        cls.info = make_samples(cls.tmp / "samples")
        cls.files = cls.info["files"]
        cls.out_dir = cls.tmp / "out"
        cls.out_dir.mkdir()
        # Force tiny chunks so multi-chunk paths are exercised.
        cls._old_chunk = config.TIME_CHUNK
        config.TIME_CHUNK = 2

    @classmethod
    def tearDownClass(cls):
        config.TIME_CHUNK = cls._old_chunk
        shutil.rmtree(cls.tmp, ignore_errors=True)
        # The shared DATA_HOME may still be used by test_jobs in the same
        # process; leave cleanup to the process exit / the last test module.
        if "NC_TEST_HOME" not in os.environ:
            shutil.rmtree(cls.home, ignore_errors=True)

    def _out_path(self, name):
        return str(self.out_dir / name)

    def test_01_normalize_lon(self):
        np.testing.assert_array_equal(
            normalize_lon(np.array([0.0, 10.0, 179.0, 180.0, 350.0])),
            np.array([0.0, 10.0, 179.0, -180.0, -10.0]))

    def test_02_success_dateline_reversed_lat_missing_merge(self):
        """Lon box [170,-170] crosses the date line and excludes the conflict
        point at lon -160, so the run succeeds. Reversed-lat file and
        missing-only-equals-missing merge are exercised."""
        sel = Selection(T0, T0 + 3 * DAY, -10, 10, 170.0, -170.0)
        out = self._out_path("ok.nc")
        stats = run_extraction(self.files, sel, out)
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(stats["lat_count"], 3)   # -10, 0, 10
        # converted lons in the box: -180, -170, 170 (ascending); -160 excluded
        self.assertEqual(stats["lon_count"], 3)

        with nc.Dataset(out) as ds:
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
            times = ds.variables["time"][:]
            temp = ds.variables["temperature"][:]
            self.assertTrue(np.all(np.diff(lats) > 0))
            self.assertTrue(np.all(np.diff(lons) > 0))
            np.testing.assert_allclose(lons, [-180.0, -170.0, 170.0])
            # 6 unique timestamps across the three files
            self.assertEqual(times.size, 6)
            self.assertTrue(np.all(np.diff(times) > 0))
            # The missing value at (T0+DAY, lat -10, lon -180) is filled by
            # the present value from day2 => no missing cells remain there.
            self.assertFalse(np.ma.getmaskarray(temp).any())
            # Spot-check the filled value equals day2's value.
            with nc.Dataset(self.files[1]) as d2:
                expected = d2.variables["temperature"][0, 2, 18]  # lat -10, lon180
            ti = int(np.where(times == T0 + DAY)[0][0])
            li = int(np.where(lats == -10.0)[0][0])
            oi = int(np.where(lons == -180.0)[0][0])
            np.testing.assert_allclose(float(temp[ti, li, oi]),
                                       float(expected), rtol=1e-6)

    def test_03_conflict_fails_whole_job_no_artifact(self):
        """The closed box [170,-160] includes the planted mismatch at
        lon -160: the job fails and no output or partial temp file remains."""
        sel = Selection(T0, T0 + 3 * DAY, -10, 10, 170.0, -160.0)
        out = self._out_path("conflict.nc")
        with self.assertRaises(ExtractionError) as ctx:
            run_extraction(self.files, sel, out)
        self.assertIn("冲突", str(ctx.exception))
        self.assertFalse(os.path.exists(out))
        self.assertEqual(list(config.TMP_DIR.glob(".partial-*")), [])

    def test_04_non_wrapping_box(self):
        sel = Selection(T0, T0 + DAY, 0, 10, 20.0, 40.0)
        out = self._out_path("plain.nc")
        stats = run_extraction(self.files, sel, out)
        with nc.Dataset(out) as ds:
            np.testing.assert_allclose(ds.variables["lon"][:],
                                       [20.0, 30.0, 40.0])
            np.testing.assert_allclose(ds.variables["lat"][:], [0.0, 10.0])
            self.assertEqual(ds.variables["time"].size, 3)
        self.assertEqual(stats["records"], 3)

    def test_05_empty_selections(self):
        with self.assertRaises(ExtractionError):
            run_extraction(self.files,
                           Selection(T0 + 99 * DAY, T0 + 100 * DAY,
                                     -10, 10, 170.0, -170.0),
                           self._out_path("empty.nc"))
        with self.assertRaises(ExtractionError):
            run_extraction(self.files,
                           Selection(T0, T0 + DAY, 60, 70, 170.0, -170.0),
                           self._out_path("empty2.nc"))
        with self.assertRaises(ExtractionError):
            run_extraction(self.files,
                           Selection(T0, T0 + DAY, -10, 10, 51.0, 52.0),
                           self._out_path("empty3.nc"))

    def test_06_missing_equals_missing_only(self):
        """Two files both missing at the same cell merge as one missing;
        missing vs present adopts the present value."""
        # Build a tiny pair of files.
        d = self.tmp / "mm"
        d.mkdir(exist_ok=True)
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 180.0])
        fill = np.float32(-9999.0)

        def mk(name, vals, reverse=False):
            with nc.Dataset(str(d / name), "w") as ds:
                ds.createDimension("time", 1)
                ds.createDimension("lat", 2)
                ds.createDimension("lon", 2)
                ds.createVariable("time", "i8", ("time",))[:] = [T0]
                ds.createVariable("lat", "f8", ("lat",))[:] = (
                    lat[::-1] if reverse else lat)
                ds.createVariable("lon", "f8", ("lon",))[:] = lon
                v = ds.createVariable("temperature", "f4",
                                      ("time", "lat", "lon"), fill_value=fill)
                v[0] = np.ma.array(vals[::-1] if reverse else vals,
                                   mask=np.isnan(vals[::-1] if reverse
                                                 else vals))
        a = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
        b = np.array([[np.nan, 1.0], [np.nan, 3.0]], dtype=np.float32)
        mk("a.nc", a, reverse=True)
        mk("b.nc", b, reverse=False)
        out = self._out_path("mm.nc")
        run_extraction([str(d / "a.nc"), str(d / "b.nc")],
                       Selection(T0, T0, 0, 1, -180.0, 170.0), out)
        with nc.Dataset(out) as ds:
            t = ds.variables["temperature"][0]
            mask = np.ma.getmaskarray(t)
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
            # Output axes: ascending lat, converted lon ascending (-180, 0).
            np.testing.assert_allclose(lats, [0.0, 1.0])
            np.testing.assert_allclose(lons, [-180.0, 0.0])
            # Canonical cells (lat rows ascending, lon cols -180 then 0):
            #   (lat 0, -180): a present 1.0, b missing      -> 1.0
            #   (lat 0,    0): a missing, b missing          -> missing
            #   (lat 1, -180): a present 3.0, b present 3.0  -> 3.0
            #   (lat 1,    0): a present 2.0, b missing      -> 2.0
            self.assertFalse(mask[0, 0])
            self.assertAlmostEqual(float(t[0, 0]), 1.0, places=5)
            self.assertTrue(mask[0, 1])   # missing only equals missing
            self.assertFalse(mask[1, 0])
            self.assertAlmostEqual(float(t[1, 0]), 3.0, places=5)
            self.assertFalse(mask[1, 1])
            self.assertAlmostEqual(float(t[1, 1]), 2.0, places=5)

    def test_07_grid_mismatch_rejected(self):
        d = self.tmp / "grid"
        d.mkdir(exist_ok=True)
        p = d / "other.nc"
        with nc.Dataset(str(p), "w") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            ds.createVariable("time", "i8", ("time",))[:] = [T0]
            ds.createVariable("lat", "f8", ("lat",))[:] = [0.0, 9.0]
            ds.createVariable("lon", "f8", ("lon",))[:] = [0.0, 10.0]
            v = ds.createVariable("temperature", "f4",
                                  ("time", "lat", "lon"))
            v[:] = np.zeros((1, 2, 2), np.float32)
        with self.assertRaises(ExtractionError):
            validate_files([self.files[0], str(p)],
                           Selection(T0, T0, -10, 10, 170.0, -170.0))

    def test_08_reads_are_bounded_by_time_chunk(self):
        """The time dimension of every raw variable read is capped by
        TIME_CHUNK even when a file spans many more time records, proving
        memory does not grow with the total number of time records."""
        import netCDF4 as nc4
        from app import engine

        n_records = config.TIME_CHUNK * 12 + 7
        d = self.tmp / "chunk"
        d.mkdir(exist_ok=True)
        p = d / "long.nc"
        with nc.Dataset(str(p), "w") as ds:
            ds.createDimension("time", n_records)
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            ds.createVariable("time", "i8", ("time",))[:] = \
                np.arange(T0, T0 + n_records, dtype=np.int64)
            ds.createVariable("lat", "f8", ("lat",))[:] = [0.0, 1.0]
            ds.createVariable("lon", "f8", ("lon",))[:] = [0.0, 180.0]
            v = ds.createVariable("temperature", "f4",
                                  ("time", "lat", "lon"))
            v[:] = np.ones((n_records, 2, 2), np.float32)

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
            out = self._out_path("chunked.nc")
            run_extraction([str(p)],
                           Selection(T0, T0 + n_records, 0, 1,
                                     -180.0, 170.0), out)
        finally:
            nc4.Variable.__getitem__ = orig_getitem

        self.assertGreater(max_time_read, 0)
        self.assertLessEqual(max_time_read, config.TIME_CHUNK)
        with nc.Dataset(out) as ds:
            self.assertEqual(ds.dimensions["time"].size, n_records)


if __name__ == "__main__":
    unittest.main()
