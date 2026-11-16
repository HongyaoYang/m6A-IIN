#!/bin/bash
#SBATCH --job-name=IIN_ELMo_full
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_ELMo_full_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_ELMo_full_%j.err

set -euo pipefail

PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"

source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate iin_elmo_py37

cd "${PROJECT_DIR}"
mkdir -p logs

echo "Environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(command -v python)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable')
print('GPU:', torch.cuda.get_device_name(0))
"

python precompute_author_elmo_direct.py \
  --fasta tech_GT/sequences_41nt.fa \
  --output tech_GT/author_elmo_features.npy \
  --options-file pretrained/elmo_small/elmo_small_options.json \
  --weight-file pretrained/elmo_small/elmo_small_weights.hdf5 \
  --batch-size 512 \
  --flush-every 100 \
  --device cuda

python -c "
import json
import numpy as np

features = np.load(
    'tech_GT/author_elmo_features.npy',
    mmap_mode='r'
)
present = np.load(
    'tech_GT/author_elmo_features.present.npy',
    mmap_mode='r'
)

print('Feature shape:', features.shape)
print('Feature dtype:', features.dtype)
print('Valid sites:', int(present.sum()))
print('Per-site shape:', features[present][0].shape)

with open('tech_GT/author_elmo_features.npy.json') as f:
    metadata = json.load(f)

print('TF32 enabled:', metadata['cuda_tf32_enabled'])
print('cuDNN deterministic:', metadata['cudnn_deterministic'])

if int(present.sum()) != 2526215:
    raise RuntimeError('Unexpected number of ELMo sites')

if features.shape[1:] != (41, 256):
    raise RuntimeError('Unexpected ELMo feature dimensions')

print('Full ELMo cache verification: PASS')
"

ls -lh \
  tech_GT/author_elmo_features.npy \
  tech_GT/author_elmo_features.present.npy \
  tech_GT/author_elmo_features.npy.json

echo "Finish: $(date)"