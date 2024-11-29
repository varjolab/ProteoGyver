#!/bin/bash

# Start dash app
cd /proteogyver
conda activate proteogyver
redis-cli shutdown
killall celery
redis-server --daemonize yes
sleep 5
#Start cron:
cron & 
#supervisorctl start celery
celery -A app.celery_app worker --loglevel=DEBUG &
sleep 15
echo "starting dash app"
gunicorn -b 0.0.0.0:8050 app:server --log-level debug --timeout 1200 

