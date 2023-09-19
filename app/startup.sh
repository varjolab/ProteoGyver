redis-cli shutdown
killall celery
redis-server --daemonize yes
celery -A app.celery_app worker --loglevel=DEBUG &
sleep 5
python app.py
