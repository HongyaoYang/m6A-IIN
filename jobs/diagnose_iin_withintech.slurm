#!/bin/bash
#SBATCH --job-name=IIN_withintech_audit
#SBATCH --partition=cpu8358
#SBATCH --qos=26cores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_withintech_audit_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_withintech_audit_%j.err

set -euo pipefail

PROJECT_DIR=/gpfs/work/bio/hongyaoyang23/m6A_IIN
source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
python diagnose_iin_withintech.py \
  --base-dir tech_GT \
  --model-dir model_IIN/final_author_elmo \
  --min-total 10
