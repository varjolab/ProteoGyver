#!/bin/bash
./parameters_toml_to_local_paths.sh
for pid in $(ps -aux | grep python | grep "app.py" | awk -F ' ' '{print $2}'); do kill -9 $pid; done
for pid in $(ps -aux | grep python | grep celery | awk -F ' ' '{print $2}'); do kill -9 $pid; done
redis-cli shutdown
killall celery
python embedded_page_updater.py
export PG_WATCHER_DEBUG=1
redis-server --daemonize yes
CPU_COUNT=$(python resources/get_cpu_count.py parameters.toml)
SCHEDULE_FILE="./data/celerybeat-schedule.db"
mkdir -p "$(dirname "$SCHEDULE_FILE")"
rm -f "$SCHEDULE_FILE"
echo "CPU count: $CPU_COUNT"
celery -A app.celery_app worker --loglevel=DEBUG --concurrency=$CPU_COUNT --logfile ./logs/$(date +"%Y-%m-%d")_celery.log & # For app
celery -A app.celery_app beat --loglevel=DEBUG --logfile ./logs/$(date +"%Y-%m-%d")_celerybeat.log --schedule "$SCHEDULE_FILE" & # For scheduled tasks
sleep 2
echo "Starting app.py"
python app.py 