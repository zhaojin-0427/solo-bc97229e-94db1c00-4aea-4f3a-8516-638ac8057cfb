"""Generate demo NetCDF archives under an isolated sample directory.

Produces three files demonstrating every required scenario:

* day1_reversed_lat.nc — latitude axis stored in *descending* order, with a
  missing value at a timestamp shared with the second file.
* day2_overlap.nc      — ascending latitude, overlapping time record whose
  values mostly match day1 (missing-only-equals-missing / present fills) but
  contain one deliberate conflict.
* day3_clean.nc        — another day, no overlap, used for successful runs.

All files share the same lat/lon grid *set*; longitudes are stored in
[0, 360). Run with an isolated output path, e.g.:

    python -I -m scripts.make_samples /tmp/ocean_samples
"""
from __future__ import annotations

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

# Fixed base time: 2023-11-14T00:00:00Z
T0 = 1700006400
DAY = 86400
FILL = np.float32(-9999.0)


def _grid():
    lat = np.arange(-30.0, 31.0, 10.0)          # -30 .. 30, 7 points
    lon = np.arange(0.0, 360.0, 10.0)           # 0 .. 350, 36 points
    return lat, lon


def _temperature_field(t_index: int, lat2d: np.ndarray, lon2d: np.ndarray,
                       seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 20.0 + 0.3 * np.cos(np.deg2rad(lat2d)) * 4.0
    wave = 2.0 * np.sin(np.deg2rad(lon2d) + t_index)
    noise = rng.normal(0, 0.05, size=lat2d.shape).astype(np.float32)
    return (base + wave + noise).astype(np.float32)


def _write(path: Path, times: np.ndarray, lat: np.ndarray, lon: np.ndarray,
          values: np.ndarray) -> None:
    with nc.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(times))
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        vt = ds.createVariable("time", "i8", ("time",))
        vlat = ds.createVariable("lat", "f8", ("lat",))
        vlon = ds.createVariable("lon", "f8", ("lon",))
        vt.units = "seconds since 1970-01-01 00:00:00 UTC"
        vlat.units = "degrees_north"
        vlon.units = "degrees_east"
        vtemp = ds.createVariable(
            "temperature", "f4", ("time", "lat", "lon"),
            fill_value=FILL, compression="zlib", complevel=4)
        vtemp.units = "degC"
        vt[:] = times
        vlat[:] = lat
        vlon[:] = lon
        vtemp[:, :, :] = np.ma.array(values, mask=np.isnan(values),
                                     fill_value=FILL)


def make_samples(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lat, lon = _grid()
    lon2d, lat2d = np.meshgrid(lon, lat)

    # ---- day 1: reversed latitude, 3 records every 12 h ----------------
    times1 = np.array([T0, T0 + 12 * 3600, T0 + DAY], dtype=np.int64)
    vals1 = np.stack([
        _temperature_field(i, lat2d, lon2d, seed=100 + i)
        for i in range(3)
    ])
    # Missing value at the shared timestamp T0+DAY inside the demo box
    # (lat -10, lon 180 => converted -180). day2 provides a real value there.
    li = int(np.where(lat == -10.0)[0][0])
    oi = int(np.where(lon == 180.0)[0][0])
    vals1[2, li, oi] = np.nan
    path1 = out_dir / "day1_reversed_lat.nc"
    _write(path1, times1, lat[::-1].copy(), lon, vals1[:, ::-1, :])

    # ---- day 2: ascending latitude, starts at shared T0+DAY -------------
    times2 = np.array([T0 + DAY, T0 + DAY + 12 * 3600], dtype=np.int64)
    vals2 = np.stack([
        _temperature_field(2, lat2d, lon2d, seed=102),  # same field as day1 t=2
        _temperature_field(3, lat2d, lon2d, seed=103),
    ])
    # The field generator is deterministic for the same t_index/seed, so
    # overlapping values are identical except for this planted conflict:
    # lat 0, lon 200 (converted -160) at T0+DAY.
    c_lat = int(np.where(lat == 0.0)[0][0])
    c_lon = int(np.where(lon == 200.0)[0][0])
    vals2[0, c_lat, c_lon] += 5.0
    path2 = out_dir / "day2_overlap_conflict.nc"
    _write(path2, times2, lat, lon, vals2)

    # ---- day 3: clean extra day, no overlaps ---------------------------
    times3 = np.array([T0 + 2 * DAY, T0 + 2 * DAY + 12 * 3600],
                      dtype=np.int64)
    vals3 = np.stack([
        _temperature_field(4, lat2d, lon2d, seed=104),
        _temperature_field(5, lat2d, lon2d, seed=105),
    ])
    path3 = out_dir / "day3_clean.nc"
    _write(path3, times3, lat[::-1].copy(), lon, vals3[:, ::-1, :])

    return {
        "dir": str(out_dir),
        "files": [str(path1), str(path2), str(path3)],
        # Suggested selection for the Web UI:
        # date-line-crossing lon box [170, -170] (the conflict point at
        # -160 is excluded), lat [-10, 10]
        "suggested": {
            "t_start": T0,
            "t_end": T0 + 3 * DAY,
            "south": -10.0, "north": 10.0,
            "west": 170.0, "east": -170.0,
        },
        "conflict_point": {"time": T0 + DAY, "lat": 0.0,
                           "lon0360": 200.0, "lon180": -160.0},
    }


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/data")
    info = make_samples(target)
    print("样例文件已生成：")
    for f in info["files"]:
        print(" -", f)
    print("建议选区参数：", info["suggested"])
    print("冲突点：", info["conflict_point"])
