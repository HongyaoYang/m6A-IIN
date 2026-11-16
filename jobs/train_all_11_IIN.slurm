#!/bin/bash
#SBATCH --job-name=IIN_11tech
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/training/IIN_11tech_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/training/IIN_11tech_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"
BASE_DIR="${PROJECT_DIR}/tech_GT"
MODEL_DIR="${PROJECT_DIR}/model_IIN/final_author_elmo"
TECH_LOG_DIR="${PROJECT_DIR}/logs/training/final_author_elmo"
COMPLETE_DIR="${MODEL_DIR}/.complete"

EPOCHS=20
BATCH_SIZE=128
LEARNING_RATE="1e-3"
NUM_WORKERS=4
SEED=2023

# GLORI is trained first as requested; the remaining datasets are ordered
# roughly from largest to smallest. Resubmission skips completed technologies.
TECHNOLOGIES=(
  GLORI
  m6A_SAC_seq
  MeRIP_seq
  m6ACE_seq
  DART_seq
  eTAM_seq
  Dorado
  scDART_seq
  m6A_REF_seq
  Oxford_nanopore
  MAZTER_seq
)

source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python

cd "${PROJECT_DIR}"
mkdir -p "${MODEL_DIR}" "${TECH_LOG_DIR}" "${COMPLETE_DIR}"

export PYTHONUNBUFFERED=1
# The graph eigendecomposition uses many small 41 x 41 matrices; one BLAS
# thread per worker avoids nested CPU oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Python: $(command -v python)"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for formal 11-technology training")
print("GPU:", torch.cuda.get_device_name(0))
PY

REQUIRED_FILES=(
  train_IIN.py
  iin_protocol_common.py
  m6a_iin_forward_regression.py
  verify_author_faithful_IIN.py
  tech_GT/train_site.csv
  tech_GT/valid_site.csv
  tech_GT/sequences_41nt.fa
  tech_GT/sequences_41nt.excluded.tsv
  tech_GT/rnafold_structures.npy
  tech_GT/author_elmo_features.npy
  tech_GT/author_elmo_features.present.npy
)

for required_file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: required file is missing or empty: ${PROJECT_DIR}/${required_file}" >&2
    exit 1
  fi
done

for tech in "${TECHNOLOGIES[@]}"; do
  if [[ ! -s "${BASE_DIR}/${tech}.csv" ]]; then
    echo "ERROR: ground-truth file is missing or empty: ${BASE_DIR}/${tech}.csv" >&2
    exit 1
  fi
done

# This checks the audited author-faithful forward path before the long run.
python verify_author_faithful_IIN.py

validate_outputs() {
  local tech="$1"
  local checkpoint="${MODEL_DIR}/${tech}_best_model.model"
  local history="${MODEL_DIR}/${tech}_train_valid_log.csv"
  local config="${MODEL_DIR}/${tech}_run_config.json"
  local predictions="${MODEL_DIR}/${tech}_best_valid_predictions.csv"

  [[ -s "${checkpoint}" ]] || return 1
  [[ -s "${history}" ]] || return 1
  [[ -s "${config}" ]] || return 1
  [[ -s "${predictions}" ]] || return 1

  python - "${tech}" "${MODEL_DIR}" "${EPOCHS}" "${BATCH_SIZE}" \
    "${LEARNING_RATE}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from iin_protocol_common import load_checkpoint

tech = sys.argv[1]
model_dir = Path(sys.argv[2])
expected_epochs = int(sys.argv[3])
expected_batch = int(sys.argv[4])
expected_lr = float(sys.argv[5])

with (model_dir / f"{tech}_run_config.json").open() as handle:
    config = json.load(handle)

assert config["technology"] == tech
assert int(config["epochs"]) == expected_epochs
assert int(config["batch_size"]) == expected_batch
assert abs(float(config["learning_rate"]) - expected_lr) < 1e-15
assert config["loss"] == "Binomial NLL only, Total > 0"

history = pd.read_csv(model_dir / f"{tech}_train_valid_log.csv")
assert len(history) == expected_epochs
assert {"epoch", "train_binomial_nll", "valid_binomial_nll"}.issubset(history.columns)

prediction_columns = set(
    pd.read_csv(
        model_dir / f"{tech}_best_valid_predictions.csv", nrows=5
    ).columns
)
assert {"technology", "split", "site_idx", "pred"}.issubset(prediction_columns)

_, metadata = load_checkpoint(
    model_dir / f"{tech}_best_model.model", torch.device("cpu")
)
assert metadata["technology"] == tech
print(f"Verified completed artifacts: {tech}")
PY
}

completed=0
for tech in "${TECHNOLOGIES[@]}"; do
  marker="${COMPLETE_DIR}/${tech}.complete"

  # A marker is only trusted when all artifacts still pass validation.  This
  # also adopts valid outputs from an earlier run that predates the marker.
  if validate_outputs "${tech}"; then
    touch "${marker}"
    echo "SKIP: ${tech} is already complete"
    completed=$((completed + 1))
    continue
  fi

  rm -f "${marker}"
  echo "============================================================"
  echo "Training ${tech}"
  echo "Started: $(date)"
  echo "Protocol: Total>0, Binomial NLL, epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}"

  python train_IIN.py \
    --tech "${tech}" \
    --base-dir "${BASE_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --metric-min-total 10 \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --device cuda \
    2>&1 | tee "${TECH_LOG_DIR}/${tech}_${SLURM_JOB_ID}.log"

  if ! validate_outputs "${tech}"; then
    echo "ERROR: ${tech} finished but its artifacts did not pass validation" >&2
    exit 1
  fi

  touch "${marker}"
  completed=$((completed + 1))
  echo "Completed ${tech}: $(date)"
done

# Recount markers so the summary is correct after resumed runs.
completed=0
for tech in "${TECHNOLOGIES[@]}"; do
  if [[ -s "${COMPLETE_DIR}/${tech}.complete" || -e "${COMPLETE_DIR}/${tech}.complete" ]]; then
    completed=$((completed + 1))
  fi
done

echo "Completed technologies: ${completed}/11"
echo "Finish: $(date)"

if (( completed == 11 )); then
  touch "${MODEL_DIR}/ALL_11_TRAINING_COMPLETE"
  echo "ALL 11 TECHNOLOGIES FINISHED. Next stage: OOD prediction."
else
  echo "Training is incomplete. Resubmit this same Slurm file after the job exits."
fi
