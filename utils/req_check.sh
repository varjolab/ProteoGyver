#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   utils/req_check.sh [parameters.toml] [output_checklist.tsv]
#
# - Generates a path checklist via app/resources/generate_path_checks.py
# - Verifies that each required path exists
# - Exits non-zero and prints a summary if anything is missing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARAMS_TOML="${1:-${REPO_ROOT}/app/parameters.toml}"
CHECKS_FILE="${2:-${REPO_ROOT}/app/resources/path_checks.tsv}"

if [[ ! -f "${PARAMS_TOML}" ]]; then
  echo "[ERROR] parameters.toml not found at: ${PARAMS_TOML}" >&2
  exit 1
fi

PY_GEN="${REPO_ROOT}/app/resources/generate_path_checks.py"
if [[ ! -f "${PY_GEN}" ]]; then
  echo "[ERROR] generate_path_checks.py not found at: ${PY_GEN}" >&2
  exit 1
fi

echo "[INFO] Generating path checklist from: ${PARAMS_TOML}"
python "${PY_GEN}" "${PARAMS_TOML}" "${CHECKS_FILE}"

if [[ ! -f "${CHECKS_FILE}" ]]; then
  echo "[ERROR] Checklist file was not created: ${CHECKS_FILE}" >&2
  exit 1
fi

echo "[INFO] Verifying required paths listed in: ${CHECKS_FILE}"
declare -a ERRORS=()
while IFS=$'\t' read -r required_path warn_msg final_msg || [[ -n "${required_path:-}" ]]; do
  # Skip comments and empty lines
  if [[ -z "${required_path:-}" ]] || [[ "${required_path}" =~ ^# ]]; then
    continue
  fi
  # Resolve path relative to repo root unless already absolute
  if [[ "${required_path}" = /* ]]; then
    full_path="${required_path}"
  else
    full_path="${REPO_ROOT}/${required_path}"
  fi
  if [[ ! -e "${full_path}" ]]; then
    ERRORS+=("[ERROR] ${warn_msg}")
    ERRORS+=("[ERROR] Missing required path: ${full_path} (from: ${required_path})")
    ERRORS+=("[ERROR] ${final_msg}")
  fi
done < "${CHECKS_FILE}"

if (( ${#ERRORS[@]} > 0 )); then
  printf '%s\n' "${ERRORS[@]}" >&2
  echo "[FAIL] One or more required paths are missing. See errors above." >&2
  exit 1
fi

echo "[OK] All required paths verified."

