redis-server --daemonize yes
celery -A app.celery_app worker --loglevel=DEBUG
