Core pipeline for shift analysis and transferability across datasets.

## Quick Start (RQ2)

### 1) Run the Pipeline

```bash
python run_model_pipeline.py \
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

## Notes

## RQ1 Notes

- RQ1 scripts live under `RQ1/datasets/<dataset>/{covariate,conditional,concept,label}`.
- RQ1 results/logs are copied under `RQ1/datasets/<dataset>/results/`.
- `RQ1/data_manifest.csv` lists data files referenced by each RQ1 script.
