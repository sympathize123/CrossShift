#!/usr/bin/env bash
# Run xgboost 1-stage pipeline (single pass).

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate navsim

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=("K-EmoPhone" "D-1" "D-2" "D-3" "crosscheck" "studentlife" "step_count" "GLOBEM" "collegeexperience")

echo "Running xgboost 1-stage pipeline for datasets: ${DATASETS[*]}"
RESULTS_DIR="${CHI_RESULTS_DIR:-results}"

for dataset in "${DATASETS[@]}"; do
  existing_summary=$(ls "${RESULTS_DIR}/distance_figures_xgb_${dataset}"/k*/combined_results_summary.csv 2>/dev/null | head -n 1 || true)
  if [[ -n "${existing_summary}" ]]; then
    echo "==> dataset=${dataset} (skip: ${existing_summary} exists)"
    continue
  fi

  echo "==> dataset=${dataset}"
  if python "${SCRIPT_DIR}/run_model_pipeline_1stage.py" \
      --model xgb \
      --dataset "${dataset}" \
      --split_strategy temporal \
      --test_size 0.2; then
    continue
  fi

  echo "    temporal split failed; retrying with random split"
  python "${SCRIPT_DIR}/run_model_pipeline_1stage.py" \
    --model xgb \
    --dataset "${dataset}" \
    --split_strategy random \
    --test_size 0.2
done

echo "Combining per-dataset summaries into ${RESULTS_DIR}/combined_results_summary.csv"
python - <<'PY'
import os
import pandas as pd

results_dir = os.environ.get("CHI_RESULTS_DIR", "results")
datasets = ["K-EmoPhone", "D-1", "D-2", "D-3", "crosscheck", "studentlife", "step_count"]
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
