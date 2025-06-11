#!/bin/bash
sed -i 's|"/proteogyver", "cache"|"/home", "kmsaloka", "Documents", "PG_cache"|g' parameters.toml
sed -i 's|"data", "Server_input", "MS run data"|"/media","kmsaloka","Expansion","20241118_parse","ms_runs"|g' parameters.toml
sed -i 's|"data", "Server_input", "MS run data handled"|"/media","kmsaloka","Expansion","20241118_parse","ms_runs_handled"|g' parameters.toml
sed -i 's|Local debug" = false|Local debug" = true|g' parameters.toml