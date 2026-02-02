# Pipeline Summary: Data Shift Analysis for Personal Health Monitoring

## Overview
This pipeline analyzes data shift (covariate, label, conditional, concept) between users in personal health monitoring datasets and measures how these shifts relate to model transferability.

---

## Two Pipeline Variants

### 1. **1-Stage Pipeline** (`RQ2/run_model_pipeline_1stage.py`)
**Key characteristic**: Feature selection happens FIRST, then user filtering based on BOTH PRAUC and AUROC simultaneously.

### 2. **2-Stage Pipeline** (`RQ2/run_model_pipeline.py`)
**Key characteristic**: User filtering happens in TWO stages - first by PRAUC, then feature selection, then final filtering by AUROC.

---

## Complete Pipeline Workflow

### **STEP 1: Data Loading**
- Load dataset from pickle file (e.g., `stress_binary_personal-current.pkl` for D#4)
- Extract: `X` (features), `y` (labels), `groups` (user IDs), `t`/`datetimes` (timestamps)
- Dataset configurations defined in `RQ2/dataset_config.py`:
  - Feature categories: PhoneUsage, Social, Physical, Mobility, Sleep
  - Thresholds: `pra_threshold`, `auroc_threshold`, `top_k` (default: 40)

### **STEP 2: Train/Test Split**
**Split strategies supported:**
- **Random** (default): Stratified random split per user (80/20)
- **Temporal**: Time-ordered split per user (first 80% → train, last 20% → test)
- **LOSO** (Leave-One-Subject-Out): One user held out as test
- **StratifiedGroupKFold**: K-fold cross-validation with group awareness

**Output**: `X_train`, `y_train`, `groups_train`, `X_test`, `y_test`, `groups_test`

### **STEP 3: Data Normalization**
- **Per-user normalization**: Z-score normalization applied separately for each user
- Numeric columns normalized; categorical columns preserved
- Zero-variance categorical columns dropped
- Results cached to avoid recomputation

### **STEP 4: Feature Selection & User Filtering**

#### **1-Stage Pipeline:**
1. **Feature Selection FIRST** (on all users):
   - Train model on full training set
   - Select top-K features based on model importance (e.g., XGBoost feature_importances_)
   - Default: `top_k=40` (can be overridden by dataset config)

2. **User Filtering SECOND** (single pass):
   - For each user with selected features:
     - Train user-specific model
     - Evaluate on user's test data
     - Compute AUROC
   - **Filter**: Users with `AUROC >= auroc_threshold` (default: 0.65)
   - **Result**: Anchor users

#### **2-Stage Pipeline:**
1. **1st Round User Filtering** (by PRAUC):
   - For each user:
     - Train user-specific model on train data
     - Evaluate on test data
     - Compute PRAUC (Precision-Recall AUC)
   - **Filter**: Users with `PRAUC >= pra_threshold` (default: 0.65)
   - **Result**: `valid_users_1st`

2. **Feature Selection** (on filtered users):
   - Train model on filtered training set
   - Select top-K features based on model importance
   - Default: `top_k=40`

3. **2nd Round User Filtering** (by AUROC):
   - For each remaining user with selected features:
     - Train user-specific model
     - Evaluate on test data
     - Compute AUROC
   - **Filter**: Users with `AUROC > auroc_threshold` (default: 0.65)
   - **Result**: Anchor users

**Key Difference**: 1-stage filters by AUROC only; 2-stage filters by PRAUC first, then AUROC.

### **STEP 5: Distance Matrix Calculation**

For each **feature category** (PhoneUsage, Social, Physical, Mobility, Sleep):

#### **4 Types of Shift Calculated:**

1. **Covariate Shift** (P(X) changes):
   - Method: Wasserstein distance between feature distributions
   - Function: `compute_pairwise_wasserstein_matrix()`
   - Measures: How feature distributions differ between users

2. **Label Shift** (P(Y) changes):
   - Method: Wasserstein distance between label distributions
   - Function: `compute_pairwise_label_shift_matrix()`
   - Measures: How class priors differ between users

3. **Conditional Shift** (P(X|Y) changes):
   - Method: Optimal Transport Dataset Distance (OTDD) with feature importance weighting
   - Function: `compute_pairwise_conditional_matrix()`
   - For tree models: Uses raw features with importance weights
   - For deep models: Uses model embeddings (projected feature representations)
   - Measures: How feature distributions conditioned on labels differ

4. **Concept Shift** (P(Y|X) changes):
   - Method: Jensen-Shannon Divergence (JSD) on model predictions
   - Function: `compute_pairwise_concept_shift_matrix()`
   - Process:
     1. Train user-specific model for each user
     2. Predict on union of both users' data
     3. Compute JSD between prediction distributions
   - Alternative methods:
     - Clustering-based (KMeans adaptive binning)
     - k-NN based
     - DetectShift (conditional randomization test with KL divergence)

**Output**: Distance matrices (symmetric, N×N where N = number of anchor users) for each category and shift type.

**Caching**: All distance matrices cached to disk for reuse.

### **STEP 6: Transferability Analysis**

For each **anchor user** and each **shift type**:

1. **Train Anchor Model**:
   - Train model on anchor user's training data
   - Evaluate on anchor user's test data → **self-AUROC**

2. **Transfer to Other Users**:
   - For each other user:
     - Apply anchor model to other user's data (train + test)
     - Evaluate → **transfer-AUROC**
     - Compute **Δ AUROC = self-AUROC - transfer-AUROC**

3. **Regression Analysis**:
   - Fit linear regression: `Δ AUROC = β × distance + ε`
   - Constraint: `β ≥ 0` (positive slope only)
   - **Regression coefficient (slope)**: Measures how strongly distance predicts transferability loss

4. **Visualization**:
   - Scatter plot: distance vs. Δ AUROC per anchor user
   - Combined regression plot: all user pairs aggregated

**Output**: 
- Individual anchor plots: `{category}/{shift_type}/cov_anchor_{user_id}.png`
- Combined regression plots: `{category}/combined_regression/combined_{shift_type}.png`

### **STEP 7: Statistical Analysis**

#### **Per Category × Shift Type:**
- **Median regression coefficient**: Across all anchor users
- **Positive rate**: Fraction of anchors with positive slope
- **Statistical tests**:
  - Wilcoxon signed-rank test (one-sided, H1: median > 0)
  - Sign test (one-sided, H1: P(positive) > 0.5)
  - FDR correction (Benjamini-Hochberg) for multiple comparisons
- **Correlation metrics**:
  - Spearman ρ (rank correlation)
  - Pearson r (linear correlation)
  - Computed on all user pairs aggregated

#### **Per Shift Type (across categories):**
- Pooled analysis across all categories
- Same statistics as above

**Output CSV files:**
- `stats_summary_category_shift_auroc.csv`: Category × Shift breakdown
- `stats_summary_shift_auroc.csv`: Shift-level summary

### **STEP 8: Visualization**

1. **Distance Matrix Heatmaps**: For each category and shift type
2. **Boxplots**: Regression coefficients by category and shift type
   - Separate plots for raw features vs. embeddings (for deep models)
3. **Combined Regression Plots**: All user pairs aggregated

**Output directories:**
- `results/distance_figures_{model}_{dataset}/k{top_k}/`
- `results/distance_cache_{model}_{dataset}/k{top_k}/` (cached matrices)

### **STEP 9: Results Logging**

**Pipeline run log** (`results/pipeline_run_log.csv`):
- Timestamp, model, dataset, seed
- Split strategy, thresholds
- User counts (raw, filtered, anchor)
- Feature counts
- Summary statistics (median slopes, correlations)
- Paths to output files

---

## Key Differences: 1-Stage vs. 2-Stage

| Aspect | 1-Stage | 2-Stage |
|--------|---------|---------|
| **Feature Selection** | Before user filtering | After 1st user filtering |
| **User Filtering** | Single pass (AUROC only) | Two passes (PRAUC → AUROC) |
| **Filtering Criteria** | `AUROC >= threshold` | `PRAUC >= threshold` then `AUROC > threshold` |
| **Typical Result** | More anchor users retained | Fewer anchor users (more selective) |
| **Use Case** | Faster, less selective | More rigorous filtering |

---

## Model Support

**Tree Models** (parallel processing supported):
- XGBoost (`xgb`)
- LightGBM (`lgbm`)
- CatBoost (`catboost`)

**Deep Models** (sequential processing):
- MLP (`mlp`)
- TabTransformer (`tabtransformer`)
- FT-Transformer (`fttransformer`)
- NODE (`node`)
- TabResNet (`tabresnet`)

**Special handling for deep models:**
- Embedding-based conditional shift (uses model embeddings)
- Global embedding analysis (all features combined)
- Separate visualization for embedding-based results

---

## Output Structure

```
results/distance_figures_{model}_{dataset}/k{top_k}/
├── PhoneUsage/
│   ├── cov/                    # Covariate shift plots
│   ├── label/                  # Label shift plots
│   ├── cond/                   # Conditional shift plots
│   ├── concept/                # Concept shift plots
│   └── combined_regression/    # Aggregated regression plots
├── Social/
├── Physical/
├── Mobility/
├── Sleep/
├── Global_Embedding/           # (Deep models only)
│   ├── cond_emb/              # Embedding-based conditional shift
│   └── concept/                # Concept shift on embeddings
├── stats_summary_category_shift_auroc.csv
├── stats_summary_shift_auroc.csv
└── boxplot_coefficient_{model}_positive.png
```

---

## Key Parameters

- **`--model`**: Model type (xgb, lgbm, catboost, mlp, etc.)
- **`--dataset`**: Dataset key (D#1, D#2, D#3, D#4, studentlife, etc.)
- **`--split_strategy`**: random, temporal, loso, stratified_group_kfold
- **`--test_size`**: Test fraction (default: 0.2)
- **`--top_k`**: Number of features to select (default: 40)
- **`--auroc_threshold`**: AUROC threshold for anchor users (default: 0.65)
- **`--pra_threshold`**: PRAUC threshold (2-stage only, default: 0.65)
- **`--seed`**: Random seed (default: 42)

---

## Temporal Split Details

When `--split_strategy temporal` is used:
- For each user, data is sorted by timestamp
- First 80% (chronologically) → training
- Last 20% (chronologically) → testing
- Ensures temporal ordering (no data leakage from future to past)
- Requires `time_values` (datetimes or timestamps) in dataset

**Current usage**: `run_xgb_all_datasets.sh` runs temporal 80/20 split for all datasets.

---

## Regression Analysis Details

**Granularity**: Per **feature category**, not per individual feature.

**Structure**:
- One distance matrix per (category, shift_type) combination
- Distance computed using ALL features within that category together
- Regression coefficient computed per (anchor, category, shift) combination
- Example: One regression for "PhoneUsage" category (all 8 features combined), not individual regressions for each phone feature

**Regression Model**:
- Linear regression with positive constraint: `Δ AUROC = β × distance + ε`, where `β ≥ 0`
- Slope (β) measures: How much transferability decreases per unit increase in distance
- Positive slope indicates: Larger distance → larger transferability loss

---

## Statistical Testing

**Hypothesis**: H1: Regression coefficients are positive (distance negatively correlates with transferability)

**Tests**:
1. **Wilcoxon signed-rank test**: Tests if median coefficient > 0
2. **Sign test**: Tests if P(positive coefficient) > 0.5
3. **FDR correction**: Benjamini-Hochberg correction for multiple comparisons

**Correlation metrics**:
- **Spearman ρ**: Rank correlation (non-parametric)
- **Pearson r**: Linear correlation (parametric)

---

## Caching Strategy

**Normalization cache**: `results/cache_normalized/` (per dataset/split/seed)
**Distance matrix cache**: `results/distance_cache_{model}_{dataset}/k{top_k}/` (per category/shift type)
**Processed data cache**: `results/processed/{dataset}_filtered_{model}.pkl` (or `_1stage.pkl`)

All caches checked before recomputation to speed up repeated runs.

---

## Example Usage

```bash
# 1-Stage pipeline with temporal split
python RQ2/run_model_pipeline_1stage.py \
  --model xgb \
  --dataset D#4 \
  --split_strategy temporal \
  --test_size 0.2 \
  --top_k 40 \
  --auroc_threshold 0.65

# 2-Stage pipeline with random split
python RQ2/run_model_pipeline.py \
  --model xgb \
  --dataset D#4 \
  --split_strategy random \
  --test_size 0.2 \
  --top_k 40 \
  --pra_threshold 0.65 \
  --auroc_threshold 0.65

# Run all datasets with temporal split (1-stage)
bash run_xgb_all_datasets.sh
```

---

## Summary

This pipeline provides a comprehensive analysis of data shift types (covariate, label, conditional, concept) and their relationship to model transferability across users in personal health monitoring datasets. The analysis is performed per feature category, with statistical testing and visualization to identify which types of shift most strongly predict transferability loss.
