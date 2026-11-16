#!/bin/bash
#SBATCH --job-name=IIN_RNAfold
#SBATCH --partition=cpu8358
#SBATCH --qos=26cores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/IIN_RNAfold_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/IIN_RNAfold_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"

source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python

cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "RNAfold: $(command -v RNAfold)"
RNAfold --version

python precompute_rnafold.py \
  --fasta tech_GT/sequences_41nt.fa \
  --output tech_GT/rnafold_structures.npy \
  --jobs "${SLURM_CPUS_PER_TASK}" \
  --rnafold "${CONDA_PREFIX}/bin/RNAfold"

echo "Finish: $(date)"

ls -lh \
  tech_GT/rnafold_structures.npy \
  tech_GT/rnafold_structures.npy.json \
  tech_GT/rnafold_structures.npy.rnafold.stderr.log