"""Runtime configuration. All paths live under DATA_HOME so they can be
mounted as a Docker volume."""
import os
from pathlib import Path


def _data_home() -> Path:
    return Path(os.environ.get("DATA_HOME", "/data")).resolve()


DATA_HOME = _data_home()
UPLOAD_DIR = DATA_HOME / "uploads"
OUTPUT_DIR = DATA_HOME / "outputs"
TMP_DIR = DATA_HOME / "tmp"
DB_PATH = DATA_HOME / "jobs.db"

# Number of time records pulled from each input file in one read.
# Memory per worker ~ n_files * TIME_CHUNK * n_lat_sel * n_lon_sel * 8 bytes,
# independent of the total number of time records.
TIME_CHUNK = int(os.environ.get("TIME_CHUNK", "64"))

# NetCDF format for outputs (NETCDF4 supports compression and large files).
NETCDF_FORMAT = "NETCDF4"
COMPRESSION_LEVEL = 4

# Upload limits
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "2048")) * 1024 * 1024

for _d in (UPLOAD_DIR, OUTPUT_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)
