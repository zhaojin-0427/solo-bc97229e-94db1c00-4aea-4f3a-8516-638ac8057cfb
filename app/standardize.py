"""Training-period statistics and per-cell standardization.

The training data are successful extraction results (single NetCDF files with
``temperature(time, lat, lon)``).  For every grid cell the mean and the
*population* standard deviation (``ddof=0``) are computed over all time
records, ignoring missing values.  The statistics are written to a NetCDF
parameter file and later applied as ``(value - mean) / std``.

Design properties
-----------------
* Statistics and conversion both read/write in time chunks; memory depends on
  ``TIME_CHUNK`` and the grid size, never on the number of time records.
* Training records from several jobs are merged by timestamp with the same
  missing/conflict rules as the extraction engine.
* Parameter files are immutable snapshots: a conversion job keeps the file it
  captured when it was created, even if a later recompute becomes current.
* Cells without any valid training value, or with a standard deviation of
  zero, produce missing values in the standardized output.
* Output files are written to a hidden temp file and atomically renamed.
"""
from __future__ import annotations

import os
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np

from . import config, engine
from .engine import FileStream, NETCDF_LOCK, _merge_into


class StandardizationError(Exception):
    """User-facing failure of a statistics / standardization operation."""


Progress = Optional[Callable[[int, int], None]]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _array_var(ds: nc.Dataset, name: str) -> np.ndarray:
    """Read a variable into a plain NumPy array, turning masks/fill values
    into NaN."""
    if name not in ds.variables:
        raise StandardizationError(f"参数文件缺少变量 {name}")
    raw = ds.variables[name][:]
    if isinstance(raw, np.ma.MaskedArray):
        return raw.astype(np.float64).filled(np.nan)
    arr = np.asarray(raw, dtype=np.float64)
    fill = getattr(ds.variables[name], "_FillValue", None)
    if fill is not None:
        arr = arr.copy()
        arr[arr == float(fill)] = np.nan
    return arr


def _grid_of(meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical grid (ascending latitude, unique longitudes)."""
    return (np.array(meta["lat_sorted"], dtype=np.float64),
            np.array(meta["lon"][meta["lon_idx_uniq"]], dtype=np.float64))


def _check_same_grid(reference: Tuple[np.ndarray, np.ndarray],
                     other: Tuple[np.ndarray, np.ndarray],
                     label: str) -> None:
    r_lat, r_lon = reference
    o_lat, o_lon = other
    if o_lat.shape != r_lat.shape or not np.array_equal(o_lat, r_lat):
        raise StandardizationError(f"{label} 的纬度格网与训练参数不一致")
    if o_lon.shape != r_lon.shape or not np.array_equal(o_lon, r_lon):
        raise StandardizationError(f"{label} 的经度格网与训练参数不一致")


def _file_map(meta: dict) -> dict:
    return {
        "path": meta["path"],
        "fill": meta["fill"],
        "time_idx": np.arange(meta["time"].size, dtype=np.int64),
        "lat_idx": meta["canonical_lat_idx"],
        "lon_idx": meta["lon_idx_uniq"],
    }


# ---------------------------------------------------------------------------
# Chunked training reader (single file fast path, timestamp merge otherwise)
# ---------------------------------------------------------------------------

def _iter_training_chunks(metas: Sequence[dict],
                          chunk_size: int) -> Iterator[Tuple[np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray]]:
    """Yield ``(times, data, mask)`` chunks in ascending time order.

    Data is already in canonical ascending-lat / ascending-lon order.  With
    several training files, records sharing a timestamp are merged by the same
    rules as extraction (missing absorbs; equal values coalesce; conflicting
    values abort the statistics run).
    """
    maps = [_file_map(m) for m in metas]
    if len(maps) == 1:
        with FileStream(maps[0], chunk_size) as stream:
            while True:
                nxt = stream.next_slab()
                if nxt is None:
                    return
                yield nxt

    nlat, nlon = metas[0]["lat_sorted"].size, metas[0]["lon_idx_uniq"].size
    streams: List[FileStream] = []
    try:
        for fm in maps:
            streams.append(FileStream(fm, chunk_size).__enter__())

        pending_t: List[Optional[np.ndarray]] = [None] * len(streams)
        pending_d: List[Optional[np.ndarray]] = [None] * len(streams)
        pending_m: List[Optional[np.ndarray]] = [None] * len(streams)
        pending_pos = [0] * len(streams)

        def refill(i: int) -> bool:
            nxt = streams[i].next_slab()
            if nxt is None:
                pending_t[i] = None
                pending_d[i] = None
                pending_m[i] = None
                return False
            pending_t[i], pending_d[i], pending_m[i] = nxt
            pending_pos[i] = 0
            return True

        for i in range(len(streams)):
            refill(i)

        buf_t = np.empty(chunk_size, dtype=np.int64)
        buf_d = np.empty((chunk_size, nlat, nlon), dtype=np.float64)
        buf_m = np.empty((chunk_size, nlat, nlon), dtype=bool)
        used = 0

        def emit() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (buf_t[:used].copy(), buf_d[:used].copy(),
                    buf_m[:used].copy())

        while True:
            active = [i for i in range(len(streams))
                      if pending_t[i] is not None]
            if not active:
                break
            t_now = min(int(pending_t[i][pending_pos[i]]) for i in active)

            row_d = np.zeros((nlat, nlon), dtype=np.float64)
            row_m = np.ones((nlat, nlon), dtype=bool)
            for i in active:
                slab_t = pending_t[i]
                while slab_t is not None and \
                        int(slab_t[pending_pos[i]]) == t_now:
                    row = pending_pos[i]
                    _merge_into(row_d, row_m,
                                pending_d[i][row], pending_m[i][row],
                                os.path.basename(streams[i].path), t_now)
                    pending_pos[i] += 1
                    if pending_pos[i] >= slab_t.size:
                        refill(i)
                        slab_t = pending_t[i]
                    else:
                        slab_t = pending_t[i]

            buf_t[used] = t_now
            buf_d[used] = row_d
            buf_m[used] = row_m
            used += 1
            if used == chunk_size:
                yield emit()
                used = 0

        if used:
            yield emit()
    finally:
        for s in streams:
            try:
                s.__exit__(None, None, None)
            except Exception:
                pass


def _apply_valid(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean mask of usable values: not declared missing and finite."""
    valid = ~np.asarray(mask, dtype=bool)
    valid &= np.isfinite(np.asarray(data, dtype=np.float64))
    return valid


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _validate_training_grids(metas: Sequence[dict]) -> Tuple[np.ndarray,
                                                             np.ndarray]:
    if not metas:
        raise StandardizationError("至少选择一个已成功的提取作业作为训练数据")
    ref = _grid_of(metas[0])
    for m in metas[1:]:
        _check_same_grid(ref, _grid_of(m),
                         os.path.basename(m["path"]))
    return ref


def compute_parameters(training_paths: Sequence[str],
                       output_path: str,
                       progress: Progress = None,
                       chunk_size: Optional[int] = None) -> dict:
    """Compute per-cell mean and population standard deviation.

    Writes an immutable NetCDF parameter file to *output_path* via a temp file
    + atomic rename.  Two chunked passes (mean, then squared deviations) keep
    memory bounded and make the population variance numerically stable.
    """
    if not training_paths:
        raise StandardizationError("至少选择一个已成功的提取作业作为训练数据")
    chunk = int(chunk_size or config.TIME_CHUNK)
    if chunk <= 0:
        raise StandardizationError("分块大小必须为正整数")

    metas = [engine.read_file_meta(p, allow_converted_lon=True)
             for p in training_paths]
    lat, lon = _validate_training_grids(metas)
    nlat, nlon = lat.size, lon.size
    total_input = int(sum(m["time"].size for m in metas))
    if total_input == 0:
        raise StandardizationError("训练数据没有任何时间记录")

    # ---- pass 1: sum and valid count ------------------------------------
    total_sum = np.zeros((nlat, nlon), dtype=np.float64)
    count = np.zeros((nlat, nlon), dtype=np.int64)
    done = 0
    for tvals, data, mask in _iter_training_chunks(metas, chunk):
        valid = _apply_valid(data, mask)
        total_sum += np.where(valid, data,
                              0.0).sum(axis=0, dtype=np.float64)
        count += valid.sum(axis=0)
        done += int(tvals.size)
        if progress:
            # Pass 1 covers the first half of the progress range.
            progress(min(done // 2, total_input // 2), total_input)

    mean = np.full((nlat, nlon), np.nan, dtype=np.float64)
    has_values = count > 0
    mean[has_values] = total_sum[has_values] / count[has_values]

    # ---- pass 2: squared deviations -------------------------------------
    sum_sq = np.zeros((nlat, nlon), dtype=np.float64)
    done2 = 0
    for tvals, data, mask in _iter_training_chunks(metas, chunk):
        valid = _apply_valid(data, mask) & has_values
        dev = np.where(valid, data.astype(np.float64) - mean, 0.0)
        sum_sq += (dev * dev).sum(axis=0, dtype=np.float64)
        done2 += int(tvals.size)
        if progress:
            # Pass 2 covers the second half of the progress range.
            progress(min(total_input // 2 + done2 // 2, total_input),
                     total_input)

    var = np.full((nlat, nlon), np.nan, dtype=np.float64)
    var[has_values] = sum_sq[has_values] / count[has_values]
    # Guard tiny negative round-off from producing NaNs.
    var = np.where(has_values, np.maximum(var, 0.0), np.nan)
    std = np.sqrt(var)

    valid = has_values & np.isfinite(mean) & np.isfinite(std)
    out_mean = np.where(valid, mean, config.NORM_FILL).astype(np.float64)
    out_std = np.where(valid, std, config.NORM_FILL).astype(np.float64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = os.path.join(
        config.TMP_DIR, ".partial-" + os.path.basename(output_path))
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    ds_out = None
    try:
        with NETCDF_LOCK:
            ds_out = nc.Dataset(tmp_path, "w", format=config.NETCDF_FORMAT)
            ds_out.createDimension("lat", nlat)
            ds_out.createDimension("lon", nlon)
            v_lat = ds_out.createVariable("lat", "f8", ("lat",))
            v_lat.units = "degrees_north"
            v_lat.standard_name = "latitude"
            v_lon = ds_out.createVariable("lon", "f8", ("lon",))
            v_lon.units = "degrees_east"
            v_lon.standard_name = "longitude"

            v_mean = ds_out.createVariable(
                "mean", "f8", ("lat", "lon"), fill_value=config.NORM_FILL,
                compression="zlib", complevel=config.COMPRESSION_LEVEL)
            v_mean.long_name = "training-period mean temperature"

            v_std = ds_out.createVariable(
                "std", "f8", ("lat", "lon"), fill_value=config.NORM_FILL,
                compression="zlib", complevel=config.COMPRESSION_LEVEL)
            v_std.long_name = ("training-period population standard deviation "
                               "(ddof=0, missing values excluded)")

            v_count = ds_out.createVariable(
                "count", "i8", ("lat", "lon"), compression="zlib",
                complevel=config.COMPRESSION_LEVEL)
            v_count.long_name = "number of valid training values"

            v_lat[:] = lat
            v_lon[:] = lon
            v_mean[:, :] = out_mean
            v_std[:, :] = out_std
            v_count[:, :] = count
            ds_out.statistics = ("per-cell mean and population standard "
                                 "deviation (ddof=0); missing values ignored")
            ds_out.training_jobs_count = len(training_paths)
            ds_out.sync()
            ds_out.close()
            ds_out = None

        os.replace(tmp_path, output_path)
    except Exception:
        with NETCDF_LOCK:
            if ds_out is not None:
                try:
                    ds_out.close()
                except Exception:
                    pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    return {
        "output": output_path,
        "time_records": int(done),
        "lat_count": int(nlat),
        "lon_count": int(nlon),
        "valid_cells": int(valid.sum()),
        "zero_std_cells": int((has_values & (std == 0.0)).sum()),
        "missing_cells": int((~valid).sum()),
    }


# ---------------------------------------------------------------------------
# Parameter snapshot loading
# ---------------------------------------------------------------------------

def load_parameters(path: str) -> dict:
    """Load a parameter NetCDF file produced by :func:`compute_parameters`."""
    with NETCDF_LOCK:
        with nc.Dataset(path, "r") as ds:
            for name in ("lat", "lon", "mean", "std"):
                if name not in ds.variables:
                    raise StandardizationError(
                        f"参数文件缺少变量 {name}")
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            mean = _array_var(ds, "mean")
            std = _array_var(ds, "std")
            count = None
            if "count" in ds.variables:
                raw = ds.variables["count"][:]
                if isinstance(raw, np.ma.MaskedArray):
                    count = raw.filled(0).astype(np.int64)
                else:
                    count = np.asarray(raw, dtype=np.int64)
    if mean.shape != (lat.size, lon.size) or std.shape != mean.shape:
        raise StandardizationError("参数文件的均值/标准差维度与格网不一致")

    valid = np.isfinite(mean) & np.isfinite(std) & (std > 0.0)
    if count is not None:
        if count.shape != mean.shape:
            raise StandardizationError("参数文件的计数变量的维度与格网不一致")
        valid &= count > 0
    return {
        "path": path,
        "lat": lat,
        "lon": lon,
        "mean": mean,
        "std": std,
        "count": count,
        "valid": valid,
    }


def check_target_grid(target_path: str, params: dict) -> dict:
    """Validate that the target extraction result uses the parameter grid."""
    meta = engine.read_file_meta(target_path, allow_converted_lon=True)
    target_lat = np.asarray(meta["lat"], dtype=np.float64)
    target_lon = np.asarray(meta["lon"], dtype=np.float64)
    _check_same_grid((params["lat"], params["lon"]),
                     (target_lat, target_lon),
                     os.path.basename(target_path))
    if meta["time"].size == 0:
        raise StandardizationError("目标数据没有任何时间记录")
    return meta


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def run_normalization(target_path: str, params: dict,
                      output_path: str,
                      progress: Progress = None,
                      chunk_size: Optional[int] = None) -> dict:
    """Apply ``(value - mean) / std`` to a target extraction result.

    The target is read and the result written in time chunks.  Target
    time/lat/lon coordinates are preserved.  Missing target values, cells
    without a valid training value and cells with ``std == 0`` become missing.
    """
    if os.path.abspath(target_path) == os.path.abspath(output_path):
        raise StandardizationError("输出文件不能覆盖目标数据文件")
    chunk = int(chunk_size or config.TIME_CHUNK)
    if chunk <= 0:
        raise StandardizationError("分块大小必须为正整数")

    meta = check_target_grid(target_path, params)
    # Extraction results share the canonical grid: ascending latitude and
    # ascending converted longitudes.  Writing these arrays preserves the
    # target's spatio-temporal coordinates exactly.
    lat = np.asarray(meta["lat_sorted"], dtype=np.float64)
    lon = np.asarray(meta["lon"][meta["lon_idx_uniq"]], dtype=np.float64)
    nlat, nlon = lat.size, lon.size
    total = int(meta["time"].size)

    mean = np.asarray(params["mean"], dtype=np.float64)
    std = np.asarray(params["std"], dtype=np.float64)
    grid_valid = np.asarray(params["valid"], dtype=bool)
    if grid_valid.shape != (nlat, nlon):
        raise StandardizationError("参数文件的格网与目标格网不一致")

    fm = _file_map(meta)
    # A target extraction result is already in canonical order; keep its
    # original coordinate arrays for output.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = os.path.join(
        config.TMP_DIR, ".partial-" + os.path.basename(output_path))
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Preserve coordinate metadata from the target.
    with NETCDF_LOCK:
        with nc.Dataset(target_path, "r") as src:
            time_units = getattr(src.variables["time"], "units",
                                 "seconds since 1970-01-01 00:00:00 UTC")
            time_calendar = getattr(src.variables["time"], "calendar", None)
            lat_units = getattr(src.variables["lat"], "units", "degrees_north")
            lon_units = getattr(src.variables["lon"], "units", "degrees_east")

    ds_out = None
    stream = None
    written = 0
    try:
        with NETCDF_LOCK:
            ds_out = nc.Dataset(tmp_path, "w", format=config.NETCDF_FORMAT)
            ds_out.createDimension("time", None)
            ds_out.createDimension("lat", nlat)
            ds_out.createDimension("lon", nlon)

            v_time = ds_out.createVariable("time", "i8", ("time",))
            v_time.units = time_units
            if time_calendar:
                v_time.calendar = time_calendar
            v_time.long_name = "Unix epoch seconds"

            v_lat = ds_out.createVariable("lat", "f8", ("lat",))
            v_lat.units = lat_units
            v_lat.standard_name = "latitude"

            v_lon = ds_out.createVariable("lon", "f8", ("lon",))
            v_lon.units = lon_units
            v_lon.standard_name = "longitude"
            v_lon.comment = "copied from the target extraction result"

            v_temp = ds_out.createVariable(
                "temperature", config.NORM_DTYPE, ("time", "lat", "lon"),
                fill_value=config.NORM_FILL, compression="zlib",
                complevel=config.COMPRESSION_LEVEL)
            v_temp.long_name = ("temperature standardized by training-period "
                                "mean and population standard deviation")

            v_lat[:] = lat
            v_lon[:] = lon

        stream = FileStream(fm, chunk).__enter__()
        while True:
            nxt = stream.next_slab()
            if nxt is None:
                break
            tvals, data, mask = nxt
            valid = _apply_valid(data, mask)
            applicable = valid & grid_valid
            out = np.empty(data.shape, dtype=np.float64)
            num = data.astype(np.float64) - mean
            np.divide(num, std, out=out, where=applicable)
            out[~applicable] = config.NORM_FILL

            n = int(tvals.size)
            with NETCDF_LOCK:
                v_time[written:written + n] = tvals
                v_temp[written:written + n, :, :] = np.ma.array(
                    out, mask=~applicable, fill_value=config.NORM_FILL)
            written += n
            if progress:
                progress(written, total)

        with NETCDF_LOCK:
            ds_out.sync()
            ds_out.close()
            ds_out = None
            stream.__exit__(None, None, None)
            stream = None

        os.replace(tmp_path, output_path)
    except Exception:
        with NETCDF_LOCK:
            if ds_out is not None:
                try:
                    ds_out.close()
                except Exception:
                    pass
            if stream is not None:
                try:
                    stream.__exit__(None, None, None)
                except Exception:
                    pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    return {
        "output": output_path,
        "records": int(written),
        "lat_count": int(nlat),
        "lon_count": int(nlon),
        "valid_training_cells": int(grid_valid.sum()),
    }
