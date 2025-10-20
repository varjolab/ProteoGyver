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

# If checklist already has non-comment content, respect it and do not overwrite
if [[ -f "${CHECKS_FILE}" ]] && grep -qvE '^\s*#' "${CHECKS_FILE}"; then
  echo "[INFO] Existing checklist with entries detected at: ${CHECKS_FILE} (leaving as-is)"
else
  echo "[INFO] Generating path checklist from: ${PARAMS_TOML} (bash parser)"
  tmp_gen="$(mktemp)"
  awk -v OFS='\t' '
    function trim(s){ sub(/^\s+/, "", s); sub(/\s+$/, "", s); return s }
    function unquote(s){ gsub(/^"|"$/, "", s); return s }
    function join_with_slash(arr, n,   i, out){ out=""; for(i=1;i<=n;i++){ if(length(out)) out=out "/"; out = out arr[i] } return out }

    BEGIN { in_section=0 }
    {
      # strip comments starting with # (not inside quotes)
      line=$0
      gsub(/#.*$/, "", line)
    }
    match(line, /^\s*\[(.+)\]\s*$/, m) {
      in_section = (m[1] == "Data paths")
      next
    }
    in_section && match(line, /^\s*([^=]+)=\s*(.+)$/, kv) {
      key = trim(kv[1])
      val = trim(kv[2])
      # handle array of segments: ["a", "b", "c"]
      if(val ~ /^\[/){
        gsub(/^\[|\]$/, "", val)
        n=split(val, parts, /,/) 
        for(i=1;i<=n;i++){ parts[i]=trim(unquote(parts[i])) }
        path = join_with_slash(parts, n)
      } else {
        path = unquote(val)
      }
      # Emit: path, warning, final
      warn = key " is missing. Ensure it is present/mounted at the specified path."
      final = key " still not found; functionalities depending on it will fail."
      if(length(path)) print path, warn, final
    }
  ' "${PARAMS_TOML}" > "${tmp_gen}"

  if [[ ! -s "${tmp_gen}" ]]; then
    echo "[ERROR] No entries generated for checklist (empty)." >&2
    rm -f "${tmp_gen}"
    exit 1
  fi

  # If an existing checklist with only comments exists, append after comments; else create new file
  if [[ -f "${CHECKS_FILE}" ]]; then
    tmp_out="$(mktemp)"
    # Preserve existing content
    cat "${CHECKS_FILE}" > "${tmp_out}"
    # Append generated entries
    cat "${tmp_gen}" >> "${tmp_out}"
    mv "${tmp_out}" "${CHECKS_FILE}"
  else
    mv "${tmp_gen}" "${CHECKS_FILE}"
  fi
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
  echo "[CHECK] ${full_path}"
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

