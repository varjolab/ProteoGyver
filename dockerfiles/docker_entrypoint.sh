#!/bin/bash
set -e

# --- Optional Resource monitoring ---
if [ "$MONITOR_RESOURCES" = "true" ]; then
    echo "[INIT] Starting Resource monitor in background..."
    /proteogyver/utils/resource_monitoring.sh &
fi
# --- Activate environment and move to project folder ---
cd /proteogyver

# Initialize micromamba and activate env
export MAMBA_ROOT_PREFIX=/opt/conda
eval "$(micromamba shell hook -s bash)"
micromamba activate PG

# --- Ensure clean Redis and Celery state ---
redis-cli shutdown || true
killall celery || true

# --- Run the embedded page updater before app starts---
echo "[INIT] Running embedded page updater..."
python embedded_page_updater.py
CPU_COUNT=$(python /proteogyver/resources/get_cpu_count.py /proteogyver/parameters.toml)
echo "[INIT] Starting Redis server..."
redis-server --daemonize yes
sleep 5  # Give Redis time to start

# --- Start Celery worker in background ---
echo "[INIT] Starting Celery worker in background..."
# ensure schedule dir exists and reset schedule file (added)
SCHEDULE_FILE="/proteogyver/data/celerybeat-schedule.db"
mkdir -p "$(dirname "$SCHEDULE_FILE")"
rm -f "$SCHEDULE_FILE"
celery -A app.celery_app worker --loglevel=DEBUG --concurrency=$CPU_COUNT & # For app
celery -A app.celery_app beat --loglevel=DEBUG --concurrency=1 --schedule "$SCHEDULE_FILE" & # For scheduled tasks
sleep 5  # Give Celery time to start

# --- Start Dash app with Gunicorn ---
echo "[INIT] Starting Dash app..."
exec gunicorn -b 0.0.0.0:8050 app:server --log-level debug --timeout 1200 --workers $CPU_COUNT --threads $CPU_COUNT
