#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash prepare_tech_GT_on_HPC.sh \
#     /gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT.rar \
#     /gpfs/work/bio/hongyaoyang23/m6A_IIN
# The RAR already contains a top-level tech_GT/ directory.

archive_path="${1:-/gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT.rar}"
extract_parent="${2:-/gpfs/work/bio/hongyaoyang23/m6A_IIN}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${extract_parent}/tech_GT"

if [[ ! -f "${archive_path}" ]]; then
    echo "ERROR: archive not found: ${archive_path}" >&2
    exit 1
fi

mkdir -p "${extract_parent}"
echo "Archive: ${archive_path}"
echo "Extraction parent: ${extract_parent}"
df -h "${extract_parent}"

if command -v unrar >/dev/null 2>&1; then
    unrar t -idq "${archive_path}"
    unrar x -o- "${archive_path}" "${extract_parent}/"
elif command -v 7z >/dev/null 2>&1; then
    7z t "${archive_path}"
    7z x -aos "-o${extract_parent}" "${archive_path}"
elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -tf "${archive_path}" >/dev/null
    bsdtar --keep-old-files -xf "${archive_path}" -C "${extract_parent}"
else
    echo "ERROR: install/load unrar, 7z, or bsdtar on the HPC." >&2
    exit 1
fi

python "${script_dir}/inspect_tech_GT.py" \
    --base-dir "${data_dir}" \
    --report "${data_dir}/IIN_data_contract_report.json"

if command -v Rscript >/dev/null 2>&1; then
    Rscript "${script_dir}/inspect_tech_GT_rds.R" "${data_dir}" \
        > "${data_dir}/IIN_RDS_summary.txt" 2>&1 || true
fi

echo "Inspection complete: ${data_dir}/IIN_data_contract_report.json"
