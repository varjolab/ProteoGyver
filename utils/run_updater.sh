#!/bin/bash

IMAGE_NAME="pg_updater:1.0"
TODAY=$(date +%Y/%m/%d)

# Volume definitions: [ "host_path" "container_path" ]
VOLUMES=(
    "/data/PG_containers/db/PG_prod" "/proteogyver/data/db"
    "/data/PG_containers/conf/PG_prod/parameters.toml" "/proteogyver/parameters.toml"
    "/mnt/varjosalo_MS_rundata" "/proteogyver/data/Server_input"
    "/mnt/varjosalo_Server_output" "/proteogyver/data/Server_output"
    "/data/PG_containers/cache/PG_updater" "/proteogyver/cache"
    "/data/PG_containers/api_data/PG_prod" "/proteogyver/components/api_tools/api_data"
)

# Build the -v options
VOLUME_ARGS=()
for ((i=0; i<${#VOLUMES[@]}; i+=2)); do
    SRC="${VOLUMES[i]}"
    DEST="${VOLUMES[i+1]}"
    VOLUME_ARGS+=("-v" "${SRC}:${DEST}")
done

echo "[$TODAY] Running database updater container..."

docker run --rm \
    "${VOLUME_ARGS[@]}" \
    "${IMAGE_NAME}"
