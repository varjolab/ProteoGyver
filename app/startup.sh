#cd ..
#rsync -avh --exclude logs . ~/appdev
#cd ~/appdev
#source bin/activate
#cd app
for pid in $(ps -aux | grep python | grep "app.py" | awk -F ' ' '{print $2}'); do kill -9 $pid; done
for pid in $(ps -aux | grep python | grep celery | awk -F ' ' '{print $2}'); do kill -9 $pid; done
redis-cli shutdown
killall celery
redis-server --daemonize yes
celery -A app.celery_app worker --logfile ./logs/$(date +"%Y-%m-%d")_celery.log &
sleep 10
echo "Starting app.py"
python app.py 
