CrossShift codebase for shift analysis and transferability across datasets.

## Folder Structure

```
RQ1/
  datasets/            # shift-type analysis scripts
  sample_features/     # small sample feature files (reviewers)
  data_manifest.csv
RQ2/
  run_model_pipeline_1stage.py
  dataset_config.py
  models/
  utils/
  pipeline/
  sample_features/     # small sample feature files (reviewers)
  ood_detection/       # OOD detection code + figure
visual/                # visualization scripts (not tracked in git)
results/               # generated outputs (gitignored by default)
```

### Sample Feature Files (for reviewers)

To keep the anonymized repo lightweight, we include **sample extracted feature files**
for each dataset (typically a small subset of users/rows). Full feature files will be
released on the official GitHub.

- RQ1 samples: `RQ1/sample_features/<dataset>/...`
- RQ2 samples: `RQ2/sample_features/<dataset>/<data_basename>.pkl` (same basename as `RQ2/dataset_config.py`)
  - Default data mode is sample. To use full data files, set `CHI_DATA_MODE=full`.

## Quick Start (RQ2)

### Run the Pipeline (1-stage)

```bash
python RQ2/run_model_pipeline_1stage.py \
  --model xgb \
  --dataset D#4 \
  --split_strategy random \
  --test_size 0.2 \
  --pra_threshold 0.65 \
  --auroc_threshold 0.65 \
  --top_k 40
```

- Outputs go under `results/` by default.
- Override output root with `CHI_RESULTS_DIR=/path/to/results`.

## OOD Detection (RQ2)

The OOD detection code and figure referenced in the paper are provided in:
- `RQ2/ood_detection/run_ood_detection_loso.py`
- `RQ2/ood_detection/create_merged_prauc_figure_dataset.py`
- `RQ2/ood_detection/ood_detection_loso_prauc_merged_D-4.png`

## RQ1 Notes

- RQ1 scripts live under `RQ1/datasets/<dataset>/{covariate,conditional,concept,label}`.
- RQ1 results/logs are copied under `RQ1/datasets/<dataset>/results/`.
- `RQ1/data_manifest.csv` lists data files referenced by each RQ1 script.
