#!/bin/bash
#SBATCH --job-name=IIN_OOD_eval
#SBATCH --partition=cpu8358
#SBATCH --qos=26cores
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_OOD_eval_%j.out
#SBATCH --error=/gpfs/work/bio/hongyaoyang23/m6A_IIN/logs/IIN_OOD_eval_%j.err
set -euo pipefail
PROJECT_DIR="/gpfs/work/bio/hongyaoyang23/m6A_IIN"
BASE_DIR="${PROJECT_DIR}/tech_GT"
MODEL_DIR="${PROJECT_DIR}/model_IIN/final_author_elmo"
source /gpfs/work/bio/hongyaoyang23/miniforge3/etc/profile.d/conda.sh
conda activate env_python
cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Evaluation rule: Total >= 10"
python - <<'PY'
from pathlib import Path
from iin_protocol_common import TECHNOLOGIES
model_dir = Path("/gpfs/work/bio/hongyaoyang23/m6A_IIN/model_IIN/final_author_elmo")
missing = [
    tech for tech in TECHNOLOGIES
    if not (model_dir / f"{tech}_allSite_preds.csv").is_file()
    or (model_dir / f"{tech}_allSite_preds.csv").stat().st_size == 0
]
if missing:
    raise RuntimeError(f"Missing OOD prediction files: {missing}")
print(f"All {len(TECHNOLOGIES)} OOD prediction files are present")
PY
python evaluate_OOD_IIN.py \
  --base-dir "${BASE_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --min-total 10
python - <<'PY'
from pathlib import Path
import pandas as pd
model_dir = Path("/gpfs/work/bio/hongyaoyang23/m6A_IIN/model_IIN/final_author_elmo")
metrics = pd.read_csv(model_dir / "IIN_OOD_metrics.csv")
assert len(metrics) == 121, f"Expected 121 source-target rows, got {len(metrics)}"
counts = metrics["evaluation_type"].value_counts().to_dict()
assert counts.get("cross_technology_OOD") == 110, counts
assert counts.get("same_technology_holdout") == 11, counts
assert metrics["total_threshold"].eq(10).all()
print("Evaluation matrix: PASS (110 cross-technology OOD + 11 same-technology holdout)")
print("Metrics:", model_dir / "IIN_OOD_metrics.csv")
print("Summary:", model_dir / "IIN_OOD_metrics_summary.csv")
PY
echo "Finish: $(date)"
