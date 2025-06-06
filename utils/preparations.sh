BASE_DIR=${PG_DATA_DIR:-"/data"}  # Default to /data if PG_DATA_DIR not set

mkdir -p ${BASE_DIR}/PG_containers/conf/PG_prod
mkdir -p ${BASE_DIR}/PG_containers/db/PG_prod
mkdir -p ${BASE_DIR}/PG_containers/cache/PG_prod
mkdir -p ${BASE_DIR}/PG_containers/api_data/PG_prod
cd app
cp app/parameters.toml ${BASE_DIR}/PG_containers/conf/PG_prod/parameters.toml
