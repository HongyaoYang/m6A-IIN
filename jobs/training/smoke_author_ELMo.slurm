#!/bin/bash
#SBATCH --job-name=IIN_train_smoke
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/training/IIN_train_smoke_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/training/IIN_train_smoke_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"

source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python

cd "${PROJECT_DIR}"
mkdir -p logs/training smoke_author_ELMo

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
print('GPU:', torch.cuda.get_device_name(0))
"

python verify_author_faithful_IIN.py

python train_IIN.py \
  --tech MeRIP_seq \
  --model-dir smoke_author_ELMo \
  --epochs 1 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --max-train-sites 2048 \
  --max-valid-sites 512 \
  --num-workers 0 \
  --device cuda