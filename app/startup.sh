
sed -i 's|"/proteogyver", "cache"|"/home", "kmsaloka", "Documents", "PG_cache"|g' parameters.toml
sed -i 's|"/proteogyver/data/Server_input/MS run data"|"/media/kmsaloka/Expansion/20241118_parse/ms_runs/"|g' parameters.toml
sed -i 's|Local debug" = false|Local debug" = true|g' parameters.toml
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
