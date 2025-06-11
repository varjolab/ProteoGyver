#!/bin/bash
set -e

# Get absolute path to the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Now refer to files relative to the script
TARGET_FILE="$SCRIPT_DIR/../../app/parameters.toml"

echo "Reverting local paths in parameters.toml file: $TARGET_FILE"

sed -i 's|"/home", "kmsaloka", "Documents", "PG_cache"|"/proteogyver", "cache"|g' $TARGET_FILE
sed -i 's|"/media","kmsaloka","Expansion","20241118_parse","ms_runs"|"data", "Server_input", "MS run data"|g' $TARGET_FILE
sed -i 's|"/media","kmsaloka","Expansion","20241118_parse","ms_runs_handled"|"data", "Server_input", "MS run data handled"|g' $TARGET_FILE
sed -i 's|Local debug" = true|Local debug" = false|g' $TARGET_FILE
echo "Reverted local paths in parameters.toml file and added file to GIT commit: $TARGET_FILE"
git add $TARGET_FILE