# Development Log: Student C (Fatma Abu Al Wafa)
## Project 9: Neural Models for Phishing URL and Website Detection
## Queen's University | CISC 867 Deep Learning | 2026

**Role:** Student C — Baseline features, imbalanced handling, reporting  
**Notebook:** `StudentC/notebook/Student_C_baseline_models_Final_Version.ipynb`

---
## Week 1:  09 May 2026 to 12 May 2026

### Progress
- Reviewed all assigned reference papers for the classical baseline component.
- Agreed with team on 23-feature stateless design (no page rendering, no JS, no WHOIS).
- Confirmed that Student A's feature pipeline covers both character-level (raw URLs for Student B)
  and handcrafted features (for classical ML, Student C).

### Key Decisions
- **4 model families chosen:** LR, Linear SVM, RF, XGBoost.
  Rationale: covers the full linear vs. nonlinear spectrum; directly replicates prior work
  (Yahya et al. 2021, Ahammad et al. 2022) for a fair literature comparison.
- **3 imbalance strategies chosen:** no handling, class weighting, SMOTE.
  SMOTE applied to LR and SVM only; tree-based models handle imbalance through built-in
  weighting (SMOTE on RF/XGB adds noise without meaningful gain on tabular features).
- **Evaluation protocol fixed:** all models evaluated on 4 splits — validation,
  original test (~57/43), balanced test (50/50), imbalanced test (80/20).
  Using 8 metrics: accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, and confusion matrix.

### Issues Encountered
- None at this stage. Literature review and planning completed on schedule.

---

## Week 2: 13 May 2026 to 16 May 2026

### Progress
- Implemented all 10 model configurations (4 families × 3 strategies, minus SMOTE for RF/XGB).
- Designed a unified `train_and_evaluate()` function to ensure consistent metric computation
  across all models and all splits.
- Recorded CPU training times for all models.
- Wrote and submitted the midterm report (score: 98/100).

### Key Decisions
- **`RANDOM_STATE = 42`** used for all models, splits, and SMOTE sampling to ensure
  full reproducibility across runs and environments.
- **`CalibratedClassifierCV` wrapper for LinearSVC:** `LinearSVC` does not expose
  `predict_proba()`, which is required for ROC-AUC computation. Resolved by wrapping
  `LinearSVC` inside `CalibratedClassifierCV(method='sigmoid', cv=3)`. This converts
  the SVM decision function into calibrated probability estimates without changing
  classification behaviour.
- **Tree-based models use unscaled features; linear models use scaled features.**
  RF and XGB are invariant to feature scale. LR and SVM require `StandardScaler`
  (fitted on training set only, applied to validation and test — no leakage).

### Hyperparameters Fixed
| Model | Configuration |
|---|---|
| LR | `C=1.0, max_iter=2000, solver='lbfgs', RANDOM_STATE=42` |
| LR (weighted) | `class_weight='balanced'` |
| LR (SMOTE) | trained on SMOTE-balanced file (104,960 samples, 50/50) |
| RF | `n_estimators=300, min_samples_leaf=2, n_jobs=-1, RANDOM_STATE=42` |
| RF (weighted) | `class_weight='balanced'` |
| XGB | `n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, eval_metric='logloss', RANDOM_STATE=42` |
| XGB (weighted) | `scale_pos_weight = 52480/38363 = 1.3680` |
| SVM | `CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000), cv=3, method='sigmoid')` |
| SVM (weighted) | `LinearSVC(class_weight='balanced')` |
| SVM (SMOTE) | trained on SMOTE-balanced file |

### CPU Training Times (Google Colab CPU)
| Model | Time |
|---|---|
| LR (no handling) | 3.30 s |
| LR (weighted) | 2.72 s |
| LR (SMOTE) | 1.68 s |
| SVM (no handling) | 3.12 s |
| SVM (weighted) | 4.36 s |
| SVM (SMOTE) | 3.67 s |
| RF (no handling) | 41.78 s |
| RF (weighted) | 32.90 s |
| XGB (no handling) | 3.00 s |
| XGB (weighted) | 2.93 s |

### Issues Encountered
- **Issue:** `LinearSVC` raises `AttributeError: 'LinearSVC' object has no attribute 'predict_proba'`
  When the evaluation function calls `model.predict_proba(X)` for ROC-AUC.
  **Resolution:** wrapped all SVM variants in `CalibratedClassifierCV(method='sigmoid', cv=3)`,
  which adds probability calibration without requiring code changes to the evaluation function.

### Midterm Report
- Written in IEEE two-column format, official IEEE conference template.
- Submitted on schedule. Score: **98 / 100**.
- TA feedback for final report:
  1. Insert the GitHub repository link in the introduction.
  2. Add formal metric equations (Accuracy, Precision, Recall, F1, AUC, FPR, FNR).
  3. Document at least two implementation challenges.

---

## Week 3: 19 May 2026 to 22 May 2026

### Progress
- Built and trained the Stacking Ensemble.
- Selected the best model using validation F1 (no test-set access during selection).
- Completed RF feature importance analysis and feature-subset ablation study.
- Completed threshold sensitivity analysis, confusion matrices, ROC curves, error analysis,
  and CPU inference time benchmark.
- Verified all 19 output files using the notebook's final deliverables checklist.

### Key Decisions
- **Stacking Ensemble design:**
  - Base learners: RF, XGB, LR (diverse: bagging + boosting + linear).
  - Meta-learner: LR (`C=1.0, max_iter=1000`).
  - `StackingClassifier(cv=5, passthrough=False, n_jobs=-1)`.
  - `passthrough=False`: meta-learner receives only out-of-fold predictions from base
    learners, not original features. This prevents the meta-learner from overfitting
    by reducing its input dimensionality to 3 columns (one per base learner).
- **Best model selection on validation F1 only.** Test sets were not accessed during
  selection to prevent information leakage. Stacking Ensemble achieved the highest
  validation F1 = 0.9498 (validation AUC = 0.9910).
- **Threshold recommendation split by deployment context:**
  - Corporate firewall (minimise FNR): threshold 0.30–0.40, peak F1 at 0.40 = 0.9539.
  - Browser plugin (balance FPR/FNR): default threshold 0.50, F1 = 0.9519.

### Stacking Ensemble Results

**Original test set (~57/43):**

| Metric | Value |
|---|---|
| Accuracy | 0.9597 |
| Precision | 0.9584 |
| Recall | 0.9455 |
| F1-score | 0.9519 |
| ROC-AUC | 0.9918 |
| FPR | 0.0300 |
| FNR | 0.0545 |
| TN / FP / FN / TP | 10,909 / 337 / 448 / 7,773 |

**Balanced test set (50/50):**

| Metric | Value |
|---|---|
| F1-score | 0.9571 |
| ROC-AUC | 0.9918 |

**Imbalanced test set (80/20):**

| Metric | Value |
|---|---|
| F1-score | 0.9145 |
| ROC-AUC | 0.9911 |

### Feature Importance (RF Gini)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `path_length` | 0.1763 |
| 2 | `hostname_length` | 0.1262 |
| 3 | `shannon_entropy` | 0.0951 |
| 4 | `path_depth` | 0.0927 |
| 5 | `url_length` | 0.0750 |
| 6 | `num_slashes` | 0.0533 |

### Feature Ablation Study (RF, All-23 hyperparameters)

| Subset | N Features | Val F1 | Test F1 |
|---|---|---|---|
| Top-5 | 5 | 0.9145 | 0.9154 |
| Top-10 | 10 | 0.9406 | 0.9414 |
| All-23 | 23 | 0.9479 | 0.9506 |

Key finding: the top-5 features capture 96.3% of the All-23 test F1.
Adding all 23 features gains +3.52 pp. This motivated Student B's Hybrid Fusion model.

### Issues Encountered
- **Issue:** Stacking Ensemble training time on CPU was significantly longer than expected
  (339.2 seconds). This is due to 5-fold cross-validation across three base learners on a
  training set of ~90,000 samples.
  **Resolution:** No workaround was applied (cannot reduce CV below 5 without risking
  meta-learner overfitting). The 339.2 s training time is acceptable since training is
  a one-time cost; inference remains at 0.3367 ms/URL.

---
## Egypt observed a six-day official public holiday for Eid al-Adha, which ran from Tuesday, May 26, through Sunday, May 31, 2026

## Week 4: 04 June 2026 to 11 June 2026

### Progress
- Built the joint classical vs. deep learning comparison table.
- Wrote all assigned final report sections.
- Completed all Student C presentation slides.
- Confirmed all 19 deliverable files and committed to the repository.

### Key Decisions
- **Joint comparison table method:** classical results merged from `baseline_all_results.csv`;
  deep learning results read from Student B's three output CSVs
  (`results_original_test_distribution.csv`, etc...). Both datasets were aligned on the same
  column schema before concatenation.
- **TA feedback addressed:**
  1. GitHub link added to Section I.
  2. Seven formal LaTeX metric equations added to Section V.
  3. Two implementation challenges are documented in Section V.
  4. Complete The Final Report

### Final Joint Comparison (original test set, best per family)

| Model | F1-score | ms / URL | Type |
|---|---|---|---|
| CNN-FastText | 0.9938 | 0.7775 | Deep |
| Char-CNN | 0.9926 | 2.0375 | Deep |
| Hybrid Fusion-Full | 0.9926 | 1.9102 | Deep |
| CNN-BiLSTM | 0.9920 | 6.2729 | Deep |
| Stacking Ensemble | 0.9519 | 0.3367 | Classical |
| RF (no handling) | 0.9506 | 0.3631 | Classical |
| XGB (weighted) | 0.9483 | 0.0110 | Classical |

### Issues Encountered
- No blocking issues in Week 4. All sections completed on schedule.

---

## Current Status
Student C's work is complete. All classical ML results are committed to
`StudentC/Reports/Tables/` and `StudentC/Reports/Figures/`. The best baseline model
(`best_baseline_model.joblib`) is committed to `StudentC/Best_models/`.
