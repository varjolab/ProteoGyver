#!/bin/bash
set -e

# --- Optional Resource monitoring ---
if [ "$MONITOR_RESOURCES" = "true" ]; then
    echo "[INIT] Starting Resource monitor in background..."
    /proteogyver/utils/resource_monitoring.sh &
fi

# --- Activate environment and move to project folder ---
cd /proteogyver

# Source conda initialization
source /root/miniconda3/etc/profile.d/conda.sh
conda activate PG

# --- Ensure clean Redis and Celery state ---
echo "[INIT] Shutting down old Redis and Celery processes (if any)..."
redis-cli shutdown || true
killall celery || true

echo "[INIT] Starting Redis server..."
redis-server --daemonize yes
sleep 5  # Give Redis time to start

# --- Run the embedded page updater before app starts---
echo "[INIT] Running embedded page updater..."
python embedded_page_updater.py

# --- Start Celery worker in background ---
echo "[INIT] Starting Celery worker in background..."
celery -A app.celery_app worker --loglevel=DEBUG &
sleep 15

# --- Start Dash app with Gunicorn ---
echo "[INIT] Starting Dash app..."
exec gunicorn -b 0.0.0.0:8050 app:server --log-level debug --timeout 1200
