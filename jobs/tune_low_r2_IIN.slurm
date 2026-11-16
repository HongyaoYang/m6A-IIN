#!/bin/bash
#SBATCH --job-name=IIN_tune_R2
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_tune_R2_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_tune_R2_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"
BASE_DIR="${PROJECT_DIR}/tech_GT"
TUNING_DIR="${PROJECT_DIR}/model_IIN/tuning_low_r2"
LOG_DIR="${PROJECT_DIR}/logs/tuning/low_r2"
COMPLETE_DIR="${TUNING_DIR}/.complete"

MAX_EPOCHS=60
BATCH_SIZE=128
NUM_WORKERS=4
SEED=2023
GRAD_CLIP=5.0
EARLY_STOPPING_PATIENCE=15
SCHEDULER_PATIENCE=5
SCHEDULER_FACTOR=0.3
MIN_LR=1e-6

TECHNOLOGIES=(MeRIP_seq scDART_seq m6ACE_seq MAZTER_seq)
CONFIG_NAMES=(adam_3e-4 adam_1e-4 rmsprop_8e-5)
OPTIMIZERS=(adam adam rmsprop)
LEARNING_RATES=(3e-4 1e-4 8e-5)

source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python
cd "${PROJECT_DIR}"
mkdir -p "${TUNING_DIR}" "${LOG_DIR}" "${COMPLETE_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Selection rule: minimum validation Binomial NLL only"
echo "Test and OOD data are not used during tuning"

python -m py_compile train_IIN.py summarize_IIN_tuning.py
python verify_author_faithful_IIN.py

# Exercise every new optimization control on a tiny deterministic subset before
# allocating several days to the formal search.  Smoke outputs are isolated.
SMOKE_DIR="${TUNING_DIR}/smoke_${SLURM_JOB_ID}"
python train_IIN.py \
  --tech MeRIP_seq \
  --base-dir "${BASE_DIR}" \
  --model-dir "${SMOKE_DIR}" \
  --epochs 2 \
  --batch-size 128 \
  --learning-rate 3e-4 \
  --optimizer adam \
  --grad-clip "${GRAD_CLIP}" \
  --lr-scheduler plateau \
  --scheduler-factor "${SCHEDULER_FACTOR}" \
  --scheduler-patience 1 \
  --min-learning-rate "${MIN_LR}" \
  --early-stopping-patience 2 \
  --metric-min-total 10 \
  --max-train-sites 2048 \
  --max-valid-sites 512 \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device cuda

for tech in "${TECHNOLOGIES[@]}"; do
  for index in "${!CONFIG_NAMES[@]}"; do
    config_name="${CONFIG_NAMES[$index]}"
    optimizer="${OPTIMIZERS[$index]}"
    learning_rate="${LEARNING_RATES[$index]}"
    model_dir="${TUNING_DIR}/${config_name}"
    mkdir -p "${model_dir}"

    marker="${COMPLETE_DIR}/${config_name}__${tech}.complete"
    checkpoint="${model_dir}/${tech}_best_model.model"
    history="${model_dir}/${tech}_train_valid_log.csv"
    run_config="${model_dir}/${tech}_run_config.json"
    predictions="${model_dir}/${tech}_best_valid_predictions.csv"

    if [[ -e "${marker}" && -s "${checkpoint}" && -s "${history}" \
          && -s "${run_config}" && -s "${predictions}" ]]; then
      echo "SKIP complete run: ${config_name} / ${tech}"
      continue
    fi

    rm -f "${marker}"
    echo "============================================================"
    echo "Configuration: ${config_name}"
    echo "Technology: ${tech}"
    echo "Optimizer=${optimizer}, initial_lr=${learning_rate}, max_epochs=${MAX_EPOCHS}"
    echo "Started: $(date)"

    python train_IIN.py \
      --tech "${tech}" \
      --base-dir "${BASE_DIR}" \
      --model-dir "${model_dir}" \
      --epochs "${MAX_EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --learning-rate "${learning_rate}" \
      --optimizer "${optimizer}" \
      --weight-decay 0 \
      --grad-clip "${GRAD_CLIP}" \
      --lr-scheduler plateau \
      --scheduler-factor "${SCHEDULER_FACTOR}" \
      --scheduler-patience "${SCHEDULER_PATIENCE}" \
      --min-learning-rate "${MIN_LR}" \
      --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
      --metric-min-total 10 \
      --num-workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --device cuda \
      2>&1 | tee "${LOG_DIR}/${config_name}__${tech}_${SLURM_JOB_ID}.log"

    for artifact in "${checkpoint}" "${history}" "${run_config}" "${predictions}"; do
      if [[ ! -s "${artifact}" ]]; then
        echo "ERROR: missing output after training: ${artifact}" >&2
        exit 1
      fi
    done
    touch "${marker}"
    echo "Completed: ${config_name} / ${tech} at $(date)"
    python summarize_IIN_tuning.py --tuning-dir "${TUNING_DIR}"
  done
done

python summarize_IIN_tuning.py --tuning-dir "${TUNING_DIR}"
touch "${TUNING_DIR}/TUNING_COMPLETE"
echo "Finish: $(date)"
echo "No existing final checkpoint or OOD prediction was overwritten."
