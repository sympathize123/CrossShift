CrossShift codebase for shift analysis and transferability across datasets.

## Repository Organization

- `RQ1/` — shift-type analysis scripts and data manifests.
- `RQ2/` — transferability pipelines, configs, and OOD detection code.
- `visual/` — visualization scripts (kept for reference; not tracked in git).
- `results/` — generated outputs (gitignored by default).

### Sample Feature Files (for reviewers)

To keep the anonymized repo lightweight, we include **sample extracted feature files**
for each dataset (typically a small subset of users/rows). Full feature files will be
released on the official GitHub.

- RQ1 samples: `RQ1/sample_features/<dataset>/...`
- RQ2 samples: `RQ2/sample_features/<dataset>/features_sample.pkl`

## Quick Start (RQ2)

### 1) Run the Pipeline

```bash
python RQ2/run_model_pipeline.py \
  --model xgb \
  --dataset D#4 \
  --split_strategy random \
  --test_size 0.2 \
  --pra_threshold 0.65 \
  --auroc_threshold 0.65 \
  --top_k 40
```

- Outputs and caches go under `results/` by default.
- Override output root with `CHI_RESULTS_DIR=/path/to/results`.

## OOD Detection (RQ2)

The OOD detection code and figure referenced in the paper are provided in:
- `RQ2/ood_detection/run_ood_detection_loso.py`
- `RQ2/ood_detection/create_merged_prauc_figure_dataset.py`
- `RQ2/ood_detection/ood_detection_loso_prauc_merged_D-4.png`

## Notes

## RQ1 Notes

- RQ1 scripts live under `RQ1/datasets/<dataset>/{covariate,conditional,concept,label}`.
- RQ1 results/logs are copied under `RQ1/datasets/<dataset>/results/`.
- `RQ1/data_manifest.csv` lists data files referenced by each RQ1 script.
