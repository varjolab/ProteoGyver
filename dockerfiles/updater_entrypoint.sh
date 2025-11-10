#!/usr/bin/env bash
set -euo pipefail

cd /proteogyver
# Ensure config/parameters.toml exists, copy from default if not
if [ ! -f config/parameters.toml ]; then
    echo "[INIT] config/parameters.toml not found, copying default parameters.toml to config/parameters.toml"
    mkdir -p config
    cp parameters.toml config/parameters.toml
fi


args=()
if [[ "${FORCE_PG_DB_UPDATE:-0}" == "1" ]]; then
  args+=("--force-full-update")
fi

exec python /proteogyver/database_admin.py "${args[@]}" "$@"


