# Student C — Classical Machine Learning Baselines
## Project 9: Neural Models for Phishing URL and Website Detection
**Queen's University | CISC 867 Deep Learning | 2026**

**Student:** Fatma Abu Al Wafa  
**Role:** Student C — Baseline features, imbalanced handling, reporting

---

## Problem

Phishing URLs mimic legitimate websites to steal user credentials. This component
implements classical machine learning baselines that classify URLs as phishing (1) or
legitimate (0) using 23 handcrafted URL-structure features. No page rendering,
JavaScript execution, or WHOIS queries are required — the classifier is fully
stateless and suitable for real-time deployment in browser plugins and firewalls.

**Project objectives covered by Student C:**
- Implement classical feature-based baselines (LR, SVM, RF, XGBoost, Stacking Ensemble).
- Compare three imbalance-handling strategies: no handling, class weighting, SMOTE.
- Evaluate on original, balanced (50/50), and imbalanced (80/20) test distributions.
- Analyse false positives and false negatives.
- Benchmark CPU inference latency.
- Produce joint classical + deep learning comparison table.

---

## Method

**Model families:**

| Model | Type | Notes |
|---|---|---|
| Logistic Regression (LR) | Linear | Probabilistic, fast, interpretable |
| Linear SVM | Linear | Margin-based; calibrated for probability output |
| Random Forest (RF) | Nonlinear | Bagging ensemble; built-in feature importance |
| XGBoost (XGB) | Nonlinear | Gradient-boosted ensemble; fast inference |
| Stacking Ensemble | Meta-learner | RF + XGB + LR base; LR meta; best overall F1 |

**Imbalance strategies:**

| Strategy | Applied to |
|---|---|
| No handling | All 4 families |
| Class weighting (`balanced` / `scale_pos_weight`) | All 4 families |
| SMOTE oversampling | LR and SVM only |

**Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, FPR, FNR,
TN / FP / FN / TP (across 4 splits: validation, original test, balanced test, imbalanced test).

**Extended analyses:**
- Random Forest Gini feature importance (23 features).
- Feature-subset ablation: Top-5, Top-10, All-23.
- Threshold sensitivity: thresholds 0.10 – 0.90 (Precision / Recall / F1 / FNR).
- Error analysis: false positive and false negative URL length distributions.
- CPU inference benchmark: ms per URL on 1,000 test samples.

---

## Data

**Source:** Mendeley 2026 Phishing URL Dataset  
**DOI:** `10.17632/3jddhy2f6s.1`  
**Labels:** `0` = legitimate, `1` = phishing

This notebook does **not** load the raw dataset. It loads the preprocessed feature files
produced by Student A's pipeline (`StudentA/notebook`).

**Split sizes used:**

| Split | Rows | Phishing % |
|---|---|---|
| Train (original) | 90,843 | 42.2% |
| Train (SMOTE-balanced) | 104,960 | 50.0% |
| Validation | 19,466 | 42.2% |
| Test — original | 19,467 | 42.2% |
| Test — balanced 50/50 | 16,442 | 50.0% |
| Test — imbalanced 80/20 | 14,057 | 20.0% |

**23 handcrafted features used:**
`url_length`, `hostname_length`, `path_length`, `query_length`, `num_dots`,
`num_hyphens`, `num_slashes`, `num_at`, `num_question_marks`, `num_equals`,
`num_percent`, `num_digits`, `digit_ratio`, `has_https`, `has_ip`,
`subdomain_depth`, `path_depth`, `hostname_has_hyphen`, `shannon_entropy`,
`suspicious_keyword_count`, `num_underscores`, `num_ampersands`, `num_tildes`.

---

## Reproducibility

All experiments use **`RANDOM_STATE = 42`** throughout:

```python
RANDOM_STATE = 42
```

Applied to: all scikit-learn models, XGBoost, StackingClassifier internal CV folds,
and SMOTE sampling (via `random_state=RANDOM_STATE`).

**Python version:** 3.9+  
**Hardware used:** Google Colab CPU (equivalent to a standard laptop CPU, ≥8 GB RAM)  
**No GPU required.**

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Run Student A's notebook first

Student A's notebook must be executed before this notebook.
It produces all feature files that this notebook reads as input.

Notebook: `StudentA/notebook/01_studentA_data_preparation - Final Submission.ipynb`

### Step 3 — Configure paths

Open the notebook and update `BASE` in **Section 2** to point to the repository root:

```python
# Google Colab with Google Drive
BASE = Path("/content/drive/MyDrive/<your_drive_folder>/")

# Local machine
BASE = Path("/path/to/Deeplearning-URLCNN/")
```

All other paths are derived from `BASE` automatically. No other changes are needed.

### Step 4 — Verify required input files

Before running, confirm these files exist (produced by Student A):

```
StudentA/data/features/train_features.csv
StudentA/data/features/val_features.csv
StudentA/data/features/test_features.csv
StudentA/data/features/test_balanced_features.csv
StudentA/data/features/test_imbalanced_80_20_features.csv
StudentA/data/features_scaled/train_features_scaled.csv
StudentA/data/features_scaled/val_features_scaled.csv
StudentA/data/features_scaled/test_features_scaled.csv
StudentA/data/features_scaled/test_balanced_features_scaled.csv
StudentA/data/features_scaled/test_imbalanced_80_20_features_scaled.csv
StudentA/data/features_scaled/train_features_smote_scaled.csv
StudentA/data/features/test.csv                     ← raw URLs for error analysis
```

### Step 5 — (Optional) Provide Student B outputs for Section 18

Section 18 builds the joint classical + deep learning comparison table. It reads
Student B's result CSVs. If not yet available, skip Section 18 — all other sections
run independently.

```
StudentB/Reports/results_original_test_distribution.csv
StudentB/Reports/results_balanced_test_distribution.csv
StudentB/Reports/results_imbalanced_test_distribution.csv
```

### Step 6 — Run the notebook

Execute all cells sequentially. Estimated total runtime: **7–9 minutes** on CPU
(dominated by RF training ≈ 42 s and Stacking Ensemble ≈ 339 s).

| Section | Content |
|---|---|
| 1 | Imports |
| 2 | Configure paths and create output directories |
| 3 | Load preprocessed features — verify dataset shapes |
| 4 | Define unified evaluation function |
| 5 | Define models and imbalance strategies |
| 6 | Train all models (Strategies A, B, C) |
| 7 | Aggregate and save all results |
| 8 | Baseline comparison on original test set |
| 9 | Stacking Ensemble |
| 10 | Best model selection (validation F1) |
| 11 | RF feature importance analysis |
| 11.1 | Feature-subset ablation study |
| 12 | Confusion matrices across 3 test distributions |
| 13 | ROC curves — all models |
| 14 | Imbalance strategy F1 comparison |
| 15 | Best model across test distributions |
| 15.1 | Threshold sensitivity analysis |
| 16 | Error analysis (FP / FN) |
| 17 | Final summary table |
| 17.1 | CPU inference time benchmark |
| 18 | Joint classical + deep learning comparison ← requires Student B outputs |
| 19 | Final deliverables checklist |

---

## Key Results

**Best classical model: Stacking Ensemble**

| Test Distribution | F1-score | ROC-AUC | FPR | FNR |
|---|---|---|---|---|
| Original (~57/43) | **0.9519** | 0.9918 | 0.0300 | 0.0545 |
| Balanced (50/50) | 0.9571 | 0.9918 | 0.0303 | 0.0545 |
| Imbalanced (80/20) | 0.9145 | 0.9911 | 0.0300 | 0.0566 |

**Inference latency (CPU, 1,000 URLs):**

| Model | ms / URL |
|---|---|
| LR (no handling) | 0.0031 |
| XGB (weighted) | 0.0336 |
| Stacking Ensemble | 0.3367 |
| RF (no handling) | 0.3631 |

All models are below 0.4 ms/URL — suitable for real-time browser/firewall deployment.

---

## Output Files

All outputs are committed to the repository. **Do not modify these files manually.**

**Saved model (`StudentC/Best_models/`):**

| File | Description |
|---|---|
| `best_baseline_model.joblib` | Stacking Ensemble — best model by validation F1 |

**Tables (`StudentC/Reports/Tables/`):**

| File | Description |
|---|---|
| `baseline_all_results.csv` | All model × split metric results (full matrix) |
| `baseline_test_orig_comparison.csv` | Models ranked by F1 on original test set |
| `imbalance_strategy_f1_comparison.csv` | F1 by model family × strategy (pivot table) |
| `baseline_summary_for_report.csv` | Condensed 3-distribution summary |
| `feature_importance_rf.csv` | RF Gini importances for all 23 features |
| `false_positives_best_model.csv` | 337 FP URLs with predicted vs. true labels |
| `false_negatives_best_model.csv` | 448 FN URLs with predicted vs. true labels |
| `feature_ablation_results.csv` | Val F1 and Test F1 for Top-5 / Top-10 / All-23 |
| `threshold_sensitivity.csv` | Precision / Recall / F1 / FNR at 17 thresholds |
| `classical_inference_time.csv` | ms/URL for 4 models on 1,000 test URLs |

**Figures (`StudentC/Reports/Figures/`):**

| File | Description |
|---|---|
| `feature_importance_rf.png` | RF Gini importance bar chart (23 features) |
| `roc_curves_all_models.png` | ROC curves for all baseline models |
| `confusion_matrices_best_model.png` | Confusion matrices — 3 test distributions |
| `imbalance_strategy_f1_comparison_clean.png` | F1 by imbalance strategy |
| `best_f1_by_model_family_original_test.png` | Best F1 per model family |
| `best_model_across_test_sets.png` | Best model metrics across distributions |
| `error_url_length_distribution.png` | FP / FN URL length histograms |
| `threshold_sensitivity.png` | Precision–Recall–F1 vs. threshold curve |

---

## References

| # | Paper | DOI / Link |
|---|---|---|
| 1 | Le et al., "URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection," NDSS 2018 | [arXiv:1802.03162](https://arxiv.org/abs/1802.03162) |
| 2 | Adebowale et al., "Phishing Websites Detection: A Machine Learning Approach," IEEE Access 2019 | [10.1109/ACCESS.2019.2901982](https://doi.org/10.1109/ACCESS.2019.2901982) |
| 3 | Yahya et al., "Phishing URL Detection Using Machine Learning," ICoDSA 2021 | [10.1109/ICoDSA53588.2021.9617525](https://doi.org/10.1109/ICoDSA53588.2021.9617525) |
| 4 | Ahammad et al., "Phishing URL Detection Using Machine Learning with Advanced URL Feature Engineering," Adv. Eng. Softw. 2022 | [10.1016/j.advengsoft.2022.103096](https://doi.org/10.1016/j.advengsoft.2022.103096) |
