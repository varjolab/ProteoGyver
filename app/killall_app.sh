for pid in $(ps -aux | grep python | grep "app.py" | awk -F ' ' '{print $2}'); do kill -9 $pid; done
for pid in $(ps -aux | grep python | grep celery | awk -F ' ' '{print $2}'); do kill -9 $pid; done
