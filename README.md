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
Discussion/
  ood_detection/       # OOD detection code + figure
visual/                # visualization scripts (not tracked in git)
results/               # generated outputs (gitignored by default)
```

### Sample Feature Files

To keep the anonymized repo lightweight, we include **sample extracted feature files**
for each dataset (typically a small subset of users/rows). Full feature files will be
released on the official GitHub.

- RQ1 samples: `RQ1/sample_features/<dataset>/...`
- RQ2 samples: `RQ2/sample_features/<dataset>/<data_basename>.pkl` (same basename as `RQ2/dataset_config.py`)

## Datasets (Section 3.1)

User-study datasets. We analyzed three independently collected, four-week, in-the-wild user-study datasets
with 102, 112, and 115 participants, respectively. Throughout the paper, we refer to these datasets as D#1, D#2,
and D#3. Each dataset features dense per-participant ESM notifications, scheduled at 15, 12, and 10 prompts/day
(by dataset), yielding high-resolution in-situ emotion/stress labels suitable for real-time mobile stress prediction.
To minimize demographic confounds, cohorts were curated to be broadly comparable; participants are recruited
from the same institution. We publicly share the extracted features for the three collected datasets. See
Appendix A.1 for study design and collection details. Each dataset includes: (i) post-study user information; (ii)
ESM items (valence, arousal, stress, emotion duration, attention, disturbance, mental load, change checks);
(iii) smartphone logs (app usage, notifications, calls, messages, device events, key events—absent in D#1—ambient
light, physical-activity events/transitions, location, data traffic, Wi-Fi scans, and Bluetooth scans); and (iv)
wearable signals from a Fitbit Inspire HR (calories, steps, distance, heart rate). Polar H10 data were collected
for the first two datasets but are not analyzed due to inconsistent availability.

Public datasets. We additionally analyze five public in-the-wild datasets: K-EmoPhone, GLOBEM, CrossCheck,
StudentLife, and the College Experience Study. We follow the preprocessing and feature extraction procedures
from the original studies. For GLOBEM, we use a merged multi-year version; individual yearly subsets have too
few labeled samples per user for personalized modeling. Stress is the target label for all datasets except GLOBEM,
for which we use depression labels.

## Quick Start (RQ2)

### Run the Pipeline (1-stage)

```bash
python RQ2/run_model_pipeline_1stage.py \
  --model xgb \
  --dataset D-3 \
  --split_strategy random \
  --test_size 0.2 \
  --pra_threshold 0.65 \
  --auroc_threshold 0.65 \
  --top_k 40
```

- Outputs go under `results/` by default.
- Override output root with `CHI_RESULTS_DIR=/path/to/results`.

## OOD Detection (Discussion)

The OOD detection code and figure referenced in the paper are provided in:
- `Discussion/ood_detection/run_ood_detection_loso.py`
- `Discussion/ood_detection/create_merged_prauc_figure_dataset.py`
- `Discussion/ood_detection/ood_detection_loso_prauc_merged_D-3.png`

## RQ1 Notes

- RQ1 scripts live under `RQ1/datasets/<dataset>/{covariate,conditional,concept,label}`.
- RQ1 results/logs are copied under `RQ1/datasets/<dataset>/results/`.
- `RQ1/data_manifest.csv` lists data files referenced by each RQ1 script.
