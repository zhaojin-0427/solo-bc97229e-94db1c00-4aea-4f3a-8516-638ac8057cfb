"""Streaming extraction engine for daily NetCDF temperature archives.

Design properties
-----------------
* Input files share the same 1-D lat/lon grid set; lat order may be reversed;
  lon is strictly increasing in [0, 360).
* Selection: closed time interval, closed lat interval, closed lon interval
  expressed in [-180, 180). west > east means the interval crosses the
  date line.
* Output is sorted by (time, lat ascending, converted lon ascending).
* For identical (time, grid cell), equal values merge into one record;
  missing only equals missing; any other mismatch fails the whole job.
* Reading, comparing and writing happen in time chunks; memory does not grow
  with the total number of time records.
* The output file is atomically renamed into place, and a failure never leaves
  a partial result for download.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np

from . import config


class ExtractionError(Exception):
    """User-facing failure (bad file, empty selection, value conflict ...)."""


# ---------------------------------------------------------------------------
# Selection geometry
# ---------------------------------------------------------------------------

def normalize_lon(lon0360: np.ndarray) -> np.ndarray:
    """Map longitudes in [0, 360) to [-180, 180)."""
    out = np.array(lon0360, dtype=np.float64)
    out[out >= 180.0] -= 360.0
    return out


def lon_selection_indices(lon0360: np.ndarray,
                          west: float, east: float) -> np.ndarray:
    """Indices (into the original 0..360 lon axis) inside the closed interval
    [west, east] in [-180, 180) space. west > east => wraps the date line.
    Returned indices produce longitudes in ascending converted order."""
    if not (-180.0 <= west < 180.0 and -180.0 <= east < 180.0):
        raise ExtractionError("经度边界必须位于 [-180, 180) 内")
    lon = normalize_lon(lon0360)
    order = np.argsort(lon, kind="stable")
    lon_sorted = lon[order]
    if west <= east:
        hit = (lon_sorted >= west) & (lon_sorted <= east)
    else:
        hit = (lon_sorted >= west) | (lon_sorted <= east)
    sel = order[hit]
    if sel.size == 0:
        raise ExtractionError("所选经度区间内没有格点")
    return sel


def lat_selection_indices(lat: np.ndarray,
                          south: float, north: float) -> np.ndarray:
    """Indices into the lat axis for the closed interval [south, north],
    ordered by ascending latitude."""
    if south > north:
        raise ExtractionError("纬度南界不能大于北界")
    ascending = lat[1] > lat[0]
    if ascending:
        lo = int(np.searchsorted(lat, south, side="left"))
        hi = int(np.searchsorted(lat, north, side="right"))
        sel = np.arange(lo, hi, dtype=np.int64)
    else:
        # strictly decreasing: search on the reversed view
        rev = lat[::-1]
        lo = int(np.searchsorted(rev, south, side="left"))
        hi = int(np.searchsorted(rev, north, side="right"))
        sel_rev = np.arange(lo, hi, dtype=np.int64)
        sel = lat.size - 1 - sel_rev
    if sel.size == 0:
        raise ExtractionError("所选纬度区间内没有格点")
    return sel


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

REQUIRED_DIMS = ("time", "lat", "lon")
REQUIRED_VARS = ("time", "lat", "lon", "temperature")


def _read_coords(path: str) -> dict:
    with nc.Dataset(path, "r") as ds:
        for name in REQUIRED_DIMS:
            if name not in ds.dimensions:
                raise ExtractionError(f"{os.path.basename(path)} 缺少维度 {name}")
        for name in REQUIRED_VARS:
            if name not in ds.variables:
                raise ExtractionError(f"{os.path.basename(path)} 缺少变量 {name}")
        t = ds.variables["time"][:]
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        temp = ds.variables["temperature"]
        if t.ndim != 1 or lat.ndim != 1 or lon.ndim != 1:
            raise ExtractionError(f"{os.path.basename(path)} 坐标变量必须是一维")
        if temp.dimensions != ("time", "lat", "lon"):
            raise ExtractionError(
                f"{os.path.basename(path)} temperature 维度必须为 "
                "(time, lat, lon)")
        if t.size > 1 and np.any(np.diff(t.astype(np.int64)) <= 0):
            raise ExtractionError(f"{os.path.basename(path)} time 必须严格递增")
        if lat.size > 1:
            dlat = np.diff(lat)
            if not (np.all(dlat > 0) or np.all(dlat < 0)):
                raise ExtractionError(
                    f"{os.path.basename(path)} lat 必须严格单调")
        if np.any(lon < 0.0) or np.any(lon >= 360.0):
            raise ExtractionError(f"{os.path.basename(path)} lon 必须位于 [0,360)")
        if lon.size > 1 and np.any(np.diff(lon) <= 0):
            raise ExtractionError(f"{os.path.basename(path)} lon 必须严格递增")
        fill = getattr(temp, "_FillValue", None)
        if fill is None:
            fill = getattr(temp, "missing_value", None)
        return {
            "path": path,
            "time": np.array(t, dtype=np.int64),
            "lat": np.array(lat, dtype=np.float64),
            "lon": np.array(lon, dtype=np.float64),
            "dtype": temp.dtype,
            "fill": np.array(fill, dtype=temp.dtype) if fill is not None else None,
        }


@dataclass
class Selection:
    t_start: int
    t_end: int
    south: float
    north: float
    west: float
    east: float


def validate_files(paths: Sequence[str], sel: Selection) -> dict:
    """Open every file once, validate structure and shared-grid identity,
    and pre-compute per-file index maps."""
    if not paths:
        raise ExtractionError("至少需要上传一个 NetCDF 文件")
    if sel.t_start > sel.t_end:
        raise ExtractionError("时间起始不能晚于结束")

    metas = [_read_coords(p) for p in paths]
    base = metas[0]

    # Shared grid set: same number of points and identical coordinate values.
    # Lat may be stored in reverse order, so compare as sorted sets.
    base_lat_sorted = np.sort(base["lat"])
    base_lon = base["lon"]
    base_dtype = base["dtype"]
    for m in metas[1:]:
        if m["lat"].shape != base_lat_sorted.shape or \
                not np.array_equal(np.sort(m["lat"]), base_lat_sorted):
            raise ExtractionError(
                f"{os.path.basename(m['path'])} 与首个文件的纬度格点集合不一致")
        if m["lon"].shape != base_lon.shape or not np.array_equal(m["lon"], base_lon):
            raise ExtractionError(
                f"{os.path.basename(m['path'])} 与首个文件的经度格点集合不一致")
        if m["dtype"] != base_dtype:
            raise ExtractionError(
                f"{os.path.basename(m['path'])} temperature 数据类型 "
                f"{m['dtype']} 与首个文件 {base_dtype} 不一致")

    lat_idx = lat_selection_indices(base["lat"], sel.south, sel.north)
    lon_idx = lon_selection_indices(base["lon"], sel.west, sel.east)

    # Per-file maps. The canonical lat order is ascending; lon_idx already
    # yields ascending converted longitudes.
    file_maps = []
    for m in metas:
        lat_vals = m["lat"]
        # value -> storage position (order may be ascending or descending)
        pos = {float(v): i for i, v in enumerate(lat_vals)}
        canonical_lat_vals = base["lat"][lat_idx]
        f_lat_idx = np.array([pos[float(v)] for v in canonical_lat_vals],
                             dtype=np.int64)
        t = m["time"]
        i0 = int(np.searchsorted(t, sel.t_start, side="left"))
        i1 = int(np.searchsorted(t, sel.t_end, side="right"))
        file_maps.append({
            "path": m["path"],
            "fill": m["fill"],
            "time_idx": np.arange(i0, i1, dtype=np.int64),
            "lat_idx": f_lat_idx,
            "lon_idx": lon_idx,
        })

    if all(fm["time_idx"].size == 0 for fm in file_maps):
        raise ExtractionError("所选时间区间内没有任何记录")

    return {
        "metas": metas,
        "file_maps": file_maps,
        "out_lat": base["lat"][lat_idx],          # ascending
        "out_lon": normalize_lon(base["lon"][lon_idx]),  # ascending
        "dtype": base_dtype,
        "fill": base["fill"],
    }


# ---------------------------------------------------------------------------
# Chunked streaming reader
# ---------------------------------------------------------------------------

class FileStream:
    """Lazily opens a file and yields (time_value, slab) pairs in ascending
    time order, TIME_CHUNK records at a time. Slab shape is (chunk, n_lat,
    n_lon), already mapped to canonical ascending lat / converted-lon order.
    """

    def __init__(self, fm: dict, chunk_size: int):
        self.path = fm["path"]
        self.time_idx = fm["time_idx"]
        self.lat_idx = fm["lat_idx"]
        self.lon_idx = fm["lon_idx"]
        self.fill = fm["fill"]
        self.chunk_size = chunk_size
        self._ds: Optional[nc.Dataset] = None
        self._cursor = 0

    def __enter__(self) -> "FileStream":
        self._ds = nc.Dataset(self.path, "r")
        return self

    def __exit__(self, *exc) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def next_slab(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._cursor >= self.time_idx.size:
            return None
        end = min(self._cursor + self.chunk_size, self.time_idx.size)
        idx = self.time_idx[self._cursor:end]
        self._cursor = end

        tvar = self._ds.variables["time"]
        tval = np.array(tvar[idx], dtype=np.int64)
        temp = self._ds.variables["temperature"]
        # netCDF4 accepts a 1-D integer array index on the time dimension.
        # The lat/lon reordering then runs on the in-memory NumPy array with
        # np.ix_ (open mesh), which keeps the two axes independent — chained
        # fancy indexing alone would trigger cross-indexing and scramble
        # grid cells.
        raw = temp[idx, :, :]
        t0, li, oi = np.ix_(np.arange(raw.shape[0]),
                            self.lat_idx, self.lon_idx)
        slab = raw[t0, li, oi]

        if isinstance(slab, np.ma.MaskedArray):
            mask = np.ma.getmaskarray(slab)
            data = slab.data
        else:
            mask = np.zeros(slab.shape, dtype=bool)
            data = np.asarray(slab)
        # Treat the declared fill value as missing even without a mask bit.
        if self.fill is not None and not mask.all():
            mask |= (data == self.fill)
        return tval, data, mask


# ---------------------------------------------------------------------------
# Merge + conflict detection + atomic write
# ---------------------------------------------------------------------------

def _merge_into(acc_data: np.ndarray, acc_mask: np.ndarray,
                data: np.ndarray, mask: np.ndarray,
                source_label: str, t_value: int) -> None:
    """Merge one file's slab for a timestamp into the accumulator.

    Missing equals missing; missing vs present takes the present value;
    two present values must be exactly equal or the whole job fails.
    """
    both_present = ~acc_mask & ~mask
    if np.any(both_present):
        mismatch = both_present & (acc_data != data)
        if np.any(mismatch):
            pos = np.argwhere(mismatch)[0]
            raise ExtractionError(
                f"检测到冲突：time={int(t_value)}，格点 "
                f"(lat_index={int(pos[0])}, lon_index={int(pos[1])}) "
                f"存在不同数值 {acc_data[tuple(pos)]!r} != {data[tuple(pos)]!r}，"
                f"来源 {source_label}。本次任务整体失败，不生成结果文件。")
    # Missing vs present: adopt the present value.
    take = acc_mask & ~mask
    if np.any(take):
        acc_data[take] = data[take]
        acc_mask[take] = False
    # both missing => stays missing; both present & equal => unchanged.


def run_extraction(paths: Sequence[str], sel: Selection,
                   output_path: str,
                   progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Execute the full extraction. Writes to a temp file and atomically
    renames on success; on failure the temp file is removed."""
    plan = validate_files(paths, sel)
    fmaps = plan["file_maps"]
    out_lat = plan["out_lat"]
    out_lon = plan["out_lon"]
    dtype = plan["dtype"]
    fill = plan["fill"]

    if all(fm["time_idx"].size == 0 for fm in fmaps):
        raise ExtractionError("所选时间区间内没有任何记录")

    total_records = int(sum(fm["time_idx"].size for fm in fmaps))
    done_records = 0

    tmp_path = os.path.join(config.TMP_DIR,
                            ".partial-" + os.path.basename(output_path))
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    streams: List[FileStream] = []
    ds_out = None
    try:
        for fm in fmaps:
            streams.append(FileStream(fm, config.TIME_CHUNK).__enter__())

        # Per-stream pending slab state.
        pending_t: List[Optional[np.ndarray]] = [None] * len(streams)
        pending_d: List[Optional[np.ndarray]] = [None] * len(streams)
        pending_m: List[Optional[np.ndarray]] = [None] * len(streams)
        pending_pos = [0] * len(streams)

        def refill(i: int) -> bool:
            nxt = streams[i].next_slab()
            if nxt is None:
                pending_t[i] = None
                return False
            pending_t[i], pending_d[i], pending_m[i] = nxt
            pending_pos[i] = 0
            return True

        for i in range(len(streams)):
            refill(i)

        ds_out = nc.Dataset(tmp_path, "w", format=config.NETCDF_FORMAT)
        ds_out.createDimension("time", None)
        ds_out.createDimension("lat", out_lat.size)
        ds_out.createDimension("lon", out_lon.size)

        v_time = ds_out.createVariable("time", "i8", ("time",))
        v_time.units = "seconds since 1970-01-01 00:00:00 UTC"
        v_time.calendar = "proleptic_gregorian"
        v_time.long_name = "Unix epoch seconds"

        v_lat = ds_out.createVariable("lat", "f8", ("lat",))
        v_lat.units = "degrees_north"
        v_lat.standard_name = "latitude"

        v_lon = ds_out.createVariable("lon", "f8", ("lon",))
        v_lon.units = "degrees_east"
        v_lon.standard_name = "longitude"
        v_lon.comment = "converted from [0,360) to [-180,180)"

        var_kwargs = {"compression": "zlib",
                      "complevel": config.COMPRESSION_LEVEL}
        if fill is not None:
            var_kwargs["fill_value"] = fill
        v_temp = ds_out.createVariable(
            "temperature", dtype, ("time", "lat", "lon"), **var_kwargs)
        v_temp.long_name = "temperature"

        v_lat[:] = out_lat
        v_lon[:] = out_lon

        written = 0
        while True:
            active = [i for i in range(len(streams)) if pending_t[i] is not None]
            if not active:
                break
            t_now = min(int(pending_t[i][pending_pos[i]]) for i in active)

            # Accumulator for this single timestamp.
            box_shape = (out_lat.size, out_lon.size)
            acc_data = np.empty(box_shape, dtype=dtype)
            acc_mask = np.ones(box_shape, dtype=bool)

            for i in active:
                slab_t = pending_t[i]
                while slab_t is not None and \
                        int(slab_t[pending_pos[i]]) == t_now:
                    row = pending_pos[i]
                    _merge_into(acc_data, acc_mask,
                                pending_d[i][row], pending_m[i][row],
                                os.path.basename(streams[i].path), t_now)
                    pending_pos[i] += 1
                    done_records += 1
                    if pending_pos[i] >= slab_t.size:
                        refill(i)
                        slab_t = pending_t[i]
                    if progress:
                        progress(done_records, total_records)

            v_time[written] = t_now
            if fill is not None:
                out_slab = np.where(acc_mask, fill, acc_data).astype(dtype)
                v_temp[written, :, :] = np.ma.array(
                    out_slab, mask=acc_mask, fill_value=fill)
            else:
                # No declared fill: still need a mask representation; netCDF4
                # applies a default fill value for masked entries.
                v_temp[written, :, :] = np.ma.array(
                    acc_data, mask=acc_mask)
            written += 1

        ds_out.sync()
        ds_out.close()
        ds_out = None
        for s in streams:
            s.__exit__(None, None, None)
        streams.clear()

        os.replace(tmp_path, output_path)
        return {
            "records": written,
            "lat_count": int(out_lat.size),
            "lon_count": int(out_lon.size),
            "output": output_path,
        }
    except Exception:
        if ds_out is not None:
            try:
                ds_out.close()
            except Exception:
                pass
        for s in streams:
            try:
                s.__exit__(None, None, None)
            except Exception:
                pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
