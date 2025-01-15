for pid in $(ps -aux | grep python | grep "app.py" | awk -F ' ' '{print $2}'); do kill -9 $pid; done
for pid in $(ps -aux | grep python | grep celery | awk -F ' ' '{print $2}'); do kill -9 $pid; done
redis-cli shutdown
killall celery
python embedded_page_updater.py
redis-server --daemonize yes
celery -A app.celery_app worker --loglevel DEBUG --logfile ./logs/$(date +"%Y-%m-%d")_celery.log &
sleep 2
echo "Starting app.py"
python app.py 
