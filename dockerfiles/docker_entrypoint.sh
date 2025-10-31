#!/bin/bash
set -e

# --- Optional Resource monitoring ---
if [ "$MONITOR_RESOURCES" = "true" ]; then
    echo "[INIT] Starting Resource monitor in background..."
    /proteogyver/utils/resource_monitoring.sh &
fi
# --- Activate environment and move to project folder ---
cd /proteogyver
# Ensure config/parameters.toml exists, copy from default if not
if [ ! -f config/parameters.toml ]; then
    echo "[INIT] config/parameters.toml not found, copying default parameters.toml to config/parameters.toml"
    cp parameters.toml config/parameters.toml
fi

# Initialize micromamba and activate env
export MAMBA_ROOT_PREFIX=/opt/conda
eval "$(micromamba shell hook -s bash)"
micromamba activate PG

# --- Clear Python cache to prevent unmarshallable object errors ---
echo "[INIT] Clearing Python cache files..."
find /proteogyver -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /proteogyver -name "*.pyc" -delete 2>/dev/null || true
find /proteogyver -name "*.pyo" -delete 2>/dev/null || true
# Specifically clear enrichment module cache
rm -rf /proteogyver/components/enrichment/__pycache__ 2>/dev/null || true
# Clear any remaining Python cache in the environment
python -Bc "import compileall; compileall.compile_dir('/proteogyver', force=True, quiet=1)" 2>/dev/null || true

# --- Ensure clean Redis and Celery state ---
redis-cli shutdown || true
killall celery || true

# --- Run the embedded page updater before app starts---
echo "[INIT] Running embedded page updater..."
python embedded_page_updater.py
CPU_COUNT=$(python /proteogyver/resources/get_cpu_count.py /proteogyver/config/parameters.toml)
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
celery -A app.celery_app beat --loglevel=DEBUG --schedule "$SCHEDULE_FILE" & # For scheduled tasks
sleep 5  # Give Celery time to start

# --- Start Dash app with Gunicorn ---
echo "[INIT] Starting Dash app..."
exec gunicorn -b 0.0.0.0:8050 app:server --log-level debug --timeout 1200 --workers $CPU_COUNT --threads $CPU_COUNT
