#!/usr/bin/env bash
set -euo pipefail

cd /proteogyver

args=()
if [[ "${FORCE_PG_DB_UPDATE:-0}" == "1" ]]; then
  args+=("--force-full-update")
fi

exec python /proteogyver/database_admin.py "${args[@]}" "$@"


