redis-cli shutdown
killall celery
redis-server --daemonize yes
celery -A app.celery_app worker --loglevel=INFO --logfile ./logs/$(date +"%Y-%m-%d")_celery.log &
sleep 10
echo "Starting app.py"
python app.py
