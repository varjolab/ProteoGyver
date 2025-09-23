#!/bin/bash
set -e

# --- Early required path checks (two-pass) ---
CHECKS_FILE="/proteogyver/app/resources/path_checks.tsv"

# Only run if checklist file exists
if [ -f "$CHECKS_FILE" ]; then
    # Pass 1: For each missing path, display warning and block until resolved
    echo "[CHECK] Running preflight path checks (pass 1)..."
    while IFS=$'\t' read -r required_path warn_msg final_msg || [ -n "$required_path" ]; do
        # Skip comments and empty lines
        if [[ -z "$required_path" ]] || [[ "$required_path" =~ ^# ]]; then
            continue
        fi
        if [ ! -e "$required_path" ]; then
            echo "[WARN] $warn_msg"
            while [ ! -e "$required_path" ]; do
                read -rp "[ACTION] Fix the issue for: $required_path, then press Enter to re-check... " _
                if [ -e "$required_path" ]; then
                    echo "[OK] Found: $required_path"
                else
                    echo "[RECHECK] Still missing: $required_path"
                fi
            done
        fi
    done < "$CHECKS_FILE"

    # Pass 2: Re-check and display final messages for anything still missing
    echo "[CHECK] Verifying paths (pass 2)..."
    while IFS=$'\t' read -r required_path warn_msg final_msg || [ -n "$required_path" ]; do
        if [[ -z "$required_path" ]] || [[ "$required_path" =~ ^# ]]; then
            continue
        fi
        if [ ! -e "$required_path" ]; then
            echo "[NOTICE] $final_msg"
        fi
    done < "$CHECKS_FILE"
else
	echo "[CHECK] No checklist found at $CHECKS_FILE. Skipping preflight checks."
fi

# --- Optional Resource monitoring ---
if [ "$MONITOR_RESOURCES" = "true" ]; then
    echo "[INIT] Starting Resource monitor in background..."
    /proteogyver/utils/resource_monitoring.sh &
fi
# --- Activate environment and move to project folder ---
cd /proteogyver

# Source conda initialization
source /root/miniconda3/etc/profile.d/conda.sh
conda init
conda activate PG

# --- Ensure clean Redis and Celery state ---
redis-cli shutdown || true
killall celery || true

# --- Run the embedded page updater before app starts---
echo "[INIT] Running embedded page updater..."
python embedded_page_updater.py

echo "[INIT] Starting Redis server..."
redis-server --daemonize yes
sleep 5  # Give Redis time to start

# --- Start Celery worker in background ---
echo "[INIT] Starting Celery worker in background..."
celery -A app.celery_app worker --loglevel=DEBUG --concurrency=6 & # For app
celery -A app.celery_app beat --loglevel=INFO --concurrency=1 & # For cleanup
sleep 5  # Give Celery time to start

# --- Start Dash app with Gunicorn ---
echo "[INIT] Starting Dash app..."
exec gunicorn -b 0.0.0.0:8050 app:server --log-level debug --timeout 1200 --workers 4 --threads 4