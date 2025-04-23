mkdir -p /data/PG_containers/conf/PG_prod
mkdir -p /data/PG_containers/db/PG_prod
mkdir -p /data/PG_containers/cache/PG_prod
cd app
sed -i 's\"/home", "kmsaloka", "Documents", "PG_cache"\"/proteogyver", "cache"\g' parameters.toml  
sed -i 's\"Local debug" = true\"Local debug" = false\g' parameters.toml
cd ..
cp app/parameters.toml /data/PG_containers/conf/PG_prod/parameters.toml
