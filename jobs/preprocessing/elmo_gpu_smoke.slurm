#!/bin/bash
#SBATCH --job-name=IIN_ELMo_test
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_ELMo_test_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_ELMo_test_%j.err

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
import numpy
import torch
print('NumPy:', numpy.__version__)
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable')
print('GPU:', torch.cuda.get_device_name(0))
"

python precompute_author_elmo_direct.py \
  --fasta tech_GT/elmo_smoke_64.fa \
  --output tech_GT/author_elmo_smoke_gpu_notf32.npy \
  --options-file pretrained/elmo_small/elmo_small_options.json \
  --weight-file pretrained/elmo_small/elmo_small_weights.hdf5 \
  --batch-size 8 \
  --device cuda

python -c "
import numpy as np

cpu = np.load('tech_GT/author_elmo_smoke_64.npy', mmap_mode='r')
gpu = np.load(
    'tech_GT/author_elmo_smoke_gpu_notf32.npy',
    mmap_mode='r'
)

cpu_present = np.load(
    'tech_GT/author_elmo_smoke_64.present.npy'
)
gpu_present = np.load(
    'tech_GT/author_elmo_smoke_gpu_notf32.present.npy'
)

present = cpu_present & gpu_present
a = cpu[present].astype(np.float64)
b = gpu[present].astype(np.float64)
difference = np.abs(a - b)

print('Compared sites:', int(present.sum()))
print('Maximum difference:', float(difference.max()))
print('Mean absolute difference:', float(difference.mean()))
print('RMSE:', float(np.sqrt(np.mean((a-b)**2))))
print('Correlation:', float(np.corrcoef(a.ravel(), b.ravel())[0,1]))
print(
    'CPU/GPU allclose:',
    bool(np.allclose(a, b, rtol=1e-4, atol=1e-5))
)
"

echo "Finish: $(date)"