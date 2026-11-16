#!/bin/bash
#SBATCH --job-name=IIN_OOD_pred
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_OOD_pred_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_OOD_pred_%j.err
set -euo pipefail
PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"
BASE_DIR="${PROJECT_DIR}/tech_GT"
MODEL_DIR="${PROJECT_DIR}/model_IIN/final_author_elmo"
source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model directory: ${MODEL_DIR}"
if [[ ! -f "${MODEL_DIR}/ALL_11_TRAINING_COMPLETE" ]]; then
  echo "ERROR: 11-technology training completion marker is missing" >&2
  exit 1
fi
for required_file in \
  OOD_IIN.py \
  iin_protocol_common.py \
  m6a_iin_forward_regression.py \
  tech_GT/sequences_41nt.fa \
  tech_GT/rnafold_structures.npy \
  tech_GT/author_elmo_features.npy \
  tech_GT/train_site.csv \
  tech_GT/valid_site.csv; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: required file is missing or empty: ${required_file}" >&2
    exit 1
  fi
done
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for IIN OOD prediction")
print("GPU:", torch.cuda.get_device_name(0))
PY
# Predict every encodable FASTA site except each model's own training and
# validation sites. No Total threshold is applied to saved predictions.
# Existing complete outputs are checked and skipped; partial outputs resume.
python OOD_IIN.py \
  --base-dir "${BASE_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --tech all \
  --batch-size 128 \
  --chunk-size 10000 \
  --num-workers 4 \
  --device cuda
echo "All 11 OOD prediction files completed: $(date)"
echo "Next step: run evaluate_OOD_IIN.py with --min-total 10."
