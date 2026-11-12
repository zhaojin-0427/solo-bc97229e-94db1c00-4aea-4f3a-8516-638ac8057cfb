FROM python:3.11-slim

# netCDF4 wheels ship their own native libs; no system build tools needed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_HOME=/data \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY gunicorn.conf.py ./
COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /data/uploads /data/outputs /data/tmp
VOLUME ["/data"]

EXPOSE 8000

# gunicorn with a single worker: the extraction worker thread and SQLite job
# store are process-local, so all requests must be served by one process.
# Threads handle concurrent HTTP polling while a job runs in the background.
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.web:app"]
