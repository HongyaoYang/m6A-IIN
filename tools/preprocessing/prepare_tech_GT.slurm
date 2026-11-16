#!/bin/bash
#SBATCH --job-name=tech_GT_prep
#SBATCH --partition=cpu8358
#SBATCH --qos=26cores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT_prep_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/tech_GT_prep_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"
ARCHIVE="${PROJECT_DIR}/tech_GT.rar"

cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

python -c "import numpy, pandas; print('numpy:', numpy.__version__, 'pandas:', pandas.__version__)"

bash "${PROJECT_DIR}/prepare_tech_GT_on_HPC.sh" \
    "${ARCHIVE}" \
    "${PROJECT_DIR}"

echo "Finish time: $(date)"
echo "Report: ${PROJECT_DIR}/tech_GT/IIN_data_contract_report.json"