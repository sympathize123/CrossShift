#!/usr/bin/env bash
# Run xgboost 1-stage pipeline (single pass).

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate navsim

DATASETS=("D#1" "D#2" "D#3" "D#4" "crosscheck" "studentlife" "step_count")

echo "Running xgboost 1-stage pipeline for datasets: ${DATASETS[*]}"
RESULTS_DIR="${CHI_RESULTS_DIR:-results}"

for dataset in "${DATASETS[@]}"; do
  echo "==> dataset=${dataset}"
  python run_model_pipeline_1stage.py \
    --model xgb \
    --dataset "${dataset}" \
    --split_strategy temporal \
    --test_size 0.2
done

echo "Combining per-dataset summaries into ${RESULTS_DIR}/combined_results_summary.csv"
python - <<'PY'
import os
import pandas as pd

results_dir = os.environ.get("CHI_RESULTS_DIR", "results")
datasets = ["D#1", "D#2", "D#3", "D#4", "crosscheck", "studentlife", "step_count"]
rows = []
for ds in datasets:
    path = os.path.join(results_dir, f"distance_figures_xgb_{ds}", "k40", "combined_results_summary.csv")
    if not os.path.exists(path):
        print(f"[warn] missing {path}")
        continue
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df["dataset"] = ds
    rows.append(df)

if rows:
    combined = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(results_dir, "combined_results_summary.csv")
    combined.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")
else:
    print("[warn] no per-dataset summaries found")
PY

echo "All 1-stage xgboost runs completed."
