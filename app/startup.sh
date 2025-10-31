#!/bin/bash
./parameters_toml_to_local_paths.sh

# --- Clear Python cache to prevent unmarshallable object errors ---
echo "[INIT] Clearing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
# Specifically clear enrichment module cache
rm -rf components/enrichment/__pycache__ 2>/dev/null || true
# Clear any remaining Python cache in the environment
python -Bc "import compileall; compileall.compile_dir('.', force=True, quiet=1)" 2>/dev/null || true
for pid in $(ps -aux | grep python | grep "app.py" | awk -F ' ' '{print $2}'); do kill -9 $pid; done
for pid in $(ps -aux | grep python | grep celery | awk -F ' ' '{print $2}'); do kill -9 $pid; done
redis-cli shutdown
killall celery
python embedded_page_updater.py
export PG_WATCHER_DEBUG=1
redis-server --daemonize yes
CPU_COUNT=$(python resources/get_cpu_count.py config/parameters.toml)
SCHEDULE_FILE="./data/celerybeat-schedule.db"
mkdir -p "$(dirname "$SCHEDULE_FILE")"
rm -f "$SCHEDULE_FILE"
echo "CPU count: $CPU_COUNT"
celery -A app.celery_app worker --loglevel=DEBUG --concurrency=$CPU_COUNT --logfile ./logs/$(date +"%Y-%m-%d")_celery.log & # For app
celery -A app.celery_app beat --loglevel=DEBUG --logfile ./logs/$(date +"%Y-%m-%d")_celerybeat.log --schedule "$SCHEDULE_FILE" & # For scheduled tasks
sleep 2
echo "Starting app.py"
python app.py 