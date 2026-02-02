# RQ1: Shift Analysis (Covariate / Conditional / Concept / Label)

This folder contains the per-dataset shift analysis scripts and copied results.

## Layout

```
RQ1/
├── datasets/
│   └── <dataset>/
│       ├── covariate/            # covariate shift scripts (unified.py etc.)
│       ├── conditional/          # conditional shift scripts (unified.py etc.)
│       ├── concept/              # concept shift scripts (run_concept.py)
│       ├── label/                # label shift scripts (run_label.py)
│       ├── results/              # copied results/logs from NFS
│       └── Funcs/                # dataset-specific utilities (paths adjusted)
├── data/Overfitting/<dataset>/   # copied inputs/intermediate data (gitignored)
├── sample_features/<dataset>/    # small sample extracted feature files (tracked)
└── data_manifest.csv             # data file references per script
```

## Data Paths

All scripts use repo-local paths:

- `DATA_ROOT = RQ1/data/Overfitting/<dataset>`
- `RESULTS_ROOT = RQ1/datasets/<dataset>/results`

## Notes

- Original notebooks were converted to `.py` and cleaned for script execution.
- Absolute NFS paths were replaced with repo-local paths.
- Large data directories are **ignored by git**.
- Sample feature files under `RQ1/sample_features/` are small subsets for reviewers.
