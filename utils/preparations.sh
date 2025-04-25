mkdir -p /data/PG_containers/conf/PG_prod
mkdir -p /data/PG_containers/db/PG_prod
mkdir -p /data/PG_containers/cache/PG_prod
mkdir -p /data/PG_containers/api_data/PG_prod
mkdir -p /data/PG_containers/db/PG_test
mkdir -p /data/PG_containers/cache/PG_test
cd app
sed -i 's|"/home", "kmsaloka", "Documents", "PG_cache"|"/proteogyver", "cache"|g' parameters.toml
sed -i 's|"/media/kmsaloka/Expansion/20241118_parse/ms_runs/"|"/proteogyver/data/Server_input/MS run data"|g' parameters.toml
sed -i 's|Local debug" = true|Local debug" = false|g' parameters.toml
cd ..
cp app/parameters.toml /data/PG_containers/conf/PG_prod/parameters.toml
