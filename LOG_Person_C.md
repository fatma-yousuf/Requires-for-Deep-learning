# Development Log — Team 9, Project 9
# Multi-Scale Character-Level CNN and Hybrid Feature Fusion for Phishing URL Detection
# Queen's University | CISC Course | 2026

## Week 1 (5/9/26 – 5/12/26)
### Fatma (Student C) — Related Literature Review
- Reviewed Yahya et al. (ICoDSA 2021): LR, SVM, RF for phishing URL detection
- Reviewed Ahammad et al. (Adv. Eng. Softw. 2022): LR, SVM, RF, XGBoost-style baselines
  for malicious URL detection — motivates our 4-family classical ML comparison
- Key decision: 23 features extracted without page rendering, JS execution, or WHOIS queries
  → stateless, real-time deployable classifier

## Week 2 (5/13/26 – 5/15/26
### Fatma (Student C) — Classical ML Baselines Implementation
- Implemented five model families under three imbalance-handling strategies:

  Strategy A — No imbalance handling (reference baseline):
    LR (C=1.0, max_iter=2000, solver=lbfgs, RANDOM_STATE=42)
    RF (n_estimators=300, min_samples_leaf=2, n_jobs=-1, RANDOM_STATE=42)
    XGB (n_estimators=300, lr=0.1, max_depth=6, subsample=0.8, eval_metric=logloss,
         RANDOM_STATE=42)
    SVM: CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000), cv=3, method=sigmoid)

  Strategy B — Class weighting:
    LR (class_weight=balanced)
    RF (class_weight=balanced)
    XGB: scale_pos_weight = 52,480/38,363 = 1.3680 (neg/pos ratio)
    SVM: CalibratedClassifierCV(LinearSVC(class_weight=balanced), cv=3, method=sigmoid)

  Strategy C — SMOTE oversampling (LR and SVM only):
    LR and SVM trained on SMOTE-balanced training file (104,960 samples, 50/50)
    RF and XGB excluded: tree-based models handle tabular imbalance via built-in weighting

- Challenge solved: LinearSVC has no predict_proba() method required for ROC-AUC computation
  → solved using CalibratedClassifierCV with sigmoid calibration (cv=3), which converts
  SVM decision function scores into calibrated probability estimates

- Designed a unified evaluation function computing 8 metrics for every model × every split:
  accuracy, precision, recall, F1, ROC-AUC, FPR, FNR, confusion matrix values (TN/FP/FN/TP)
  Applied to: validation, original test (~57/43), balanced test (50/50), imbalanced test (80/20)

- Recorded CPU training times on Google Colab CPU:
  LR (no handling): 3.3s | LR (weighted): 2.72s | LR (SMOTE): 1.68s
  SVM (no handling): 3.12s | SVM (weighted): 4.36s | SVM (SMOTE): 3.67s
  RF (no handling): 41.78s | RF (weighted): 32.9s
  XGB (no handling): 3.0s | XGB (weighted): 2.93s

- Key finding: tree-based models (RF, XGB) achieve AUC≈0.99, F1≈0.95;
  linear models (LR, SVM) achieve AUC≈0.91, F1≈0.80 — a 15pp gap that no
  imbalance strategy can close

### All — Midterm Report (5/14/26 – 5/16/26)
- Written in IEEE two-column format using the official IEEE conference template
- All three students contributed to their respective sections
- Submitted on schedule; received score 98/100
- TA feedback: (1) insert GitHub link, (2) add formal metric equations,
  (3) document 2 implementation challenges

---

## Week 3 (5/19/26 – 5/22/26)
### Fatma (Student C) — Stacking Ensemble and Advanced Analysis
- Built Stacking Ensemble:
  Base learners: RF (n_estimators=300), XGB (n_estimators=300, lr=0.1, max_depth=6),
  LR (C=1.0, max_iter=2000)
  Meta-learner: LR (C=1.0, max_iter=1000)
  StackingClassifier(cv=5, passthrough=False, n_jobs=-1)
  passthrough=False: meta-learner receives only base-model out-of-fold predictions,
  not original features → reduces overfitting of meta-learner
  Training time: 339.2s on Google Colab CPU

- Best model selection using validation F1 only (no test-set access during selection):
  Stacking Ensemble selected: val F1=0.9498, val AUC=0.9910
  Saved as models/best_baseline_model.joblib

- Stacking Ensemble results on original test set:
  Accuracy=0.9597, Precision=0.9584, Recall=0.9455, F1=0.9519,
  AUC=0.9918, FPR=0.030, FNR=0.055
  TN=10,909 | FP=337 | FN=448 | TP=7,773

- RF Gini feature importance analysis:
  Top-6 features (red = top quartile):
  path_length (≈0.175), hostname_length (≈0.125), shannon_entropy (≈0.095),
  path_depth (≈0.093), url_length (≈0.072), num_slashes (≈0.052)
  Finding: phishing URLs use longer, more complex paths and hostnames to disguise themselves
  malicious destinations; 18 remaining features contribute < 0.06 individually

- Feature subset ablation study (RF, same hyperparameters as baseline):
  Top-5:  Val F1=0.9145, Test F1=0.9154 (captures 91.5% of All-23 test F1)
  Top-10: Val F1=0.9406, Test F1=0.9414 (+2.6pp over Top-5)
  All-23: Val F1=0.9479, Test F1=0.9506 (+3.52pp over Top-5)
  Finding: top-5 features capture most signal; adding all 23 gains +3.5pp →
  motivates the Hybrid Fusion model combining deep char features + handcrafted features

- Threshold sensitivity analysis on Stacking Ensemble (original test set):
  Threshold 0.10: Precision=0.9101, Recall=0.9822, F1=0.9448, FNR=0.0178
  Threshold 0.20: Precision=0.9320, Recall=0.9703, F1=0.9508, FNR=0.0297
  Threshold 0.30: Precision=0.9439, Recall=0.9613, F1=0.9525, FNR=0.0387
  Threshold 0.40: Precision=0.9536, Recall=0.9543, F1=0.9539, FNR=0.0457  ← peak F1
  Threshold 0.50: Precision=0.9584, Recall=0.9455, F1=0.9519, FNR=0.0545  ← default
  Threshold 0.60: Precision=0.9636, Recall=0.9364, F1=0.9498, FNR=0.0636
  Threshold 0.70: Precision=0.9677, Recall=0.9228, F1=0.9447, FNR=0.0772
  Deployment recommendations:
    Corporate firewall (minimise missed phishing): threshold 0.30–0.40
    Browser plugin (balance false alarms): threshold 0.50 (default)

- CPU inference time benchmark (1,000 test URLs, Google Colab CPU):
  LR (no handling):  0.0031 ms/URL  (108× faster than Stacking)
  XGB (weighted):    0.0336 ms/URL  (10× faster than Stacking)
  RF (no handling):  0.3631 ms/URL  (~1× comparable to Stacking)
  Stacking Ensemble: 0.3367 ms/URL  (baseline)
  All 4 models below 0.37 ms/URL → suitable for real-time browser/firewall deployment
  XGB deployment sweet spot: F1=0.9483 at 0.0336 ms/URL (–0.004 F1 vs Stacking)

- Error analysis (Stacking Ensemble, original test set):
  False Positives (n=337): legitimate URLs (20–60 chars) that mimic phishing patterns
    in structural features (high path_length, hostname complexity)
  False Negatives (n=448): phishing URLs right-skewed to 682 chars; long redirect chains
    distribute lexical signals across feature dimensions, reducing discriminability
  FN > FP: model is slightly conservative; threshold lowering directly addresses FNR

- All 19 output files verified by deliverables checklist cell (all ✓):
  Tables: baseline_all_results.csv, baseline_test_orig_comparison.csv,
    imbalance_strategy_f1_comparison.csv, baseline_summary_for_report.csv,
    feature_importance_rf.csv, false_positives_best_model.csv,
    false_negatives_best_model.csv, feature_ablation_results.csv,
    threshold_sensitivity.csv, classical_inference_time.csv
  Figures: feature_importance_rf.png, roc_curves_all_models.png,
    confusion_matrices_best_model.png, imbalance_strategy_f1_comparison_clean.png,
    best_f1_by_model_family_original_test.png, best_model_across_test_sets.png,
    error_url_length_distribution.png, threshold_sensitivity.png
  Model: best_baseline_model.joblib


## Week 4 (5/30/26 – 6/4/26
### Fatma (Student C) — Final Report and Presentation
- Completed joint classical vs deep comparison table using final_all_models_comparison.csv:
  Deep models outperform the best classical by +4.19pp F1 (CNN-FastText 0.9938 vs Stacking 0.9519)
  Classical models 2.3× faster at inference: XGB 0.01ms vs CNN-FastText 0.78ms/URL
  For CPU-constrained deployment, XGB optimal: F1=0.9483 at 0.01ms/URL

- Wrote final report sections:
  Section I (Introduction): added GitHub link per TA feedback
  Section V (Evaluation metrics): added 7 formal LaTeX equations:
    Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, Efficiency Score
    per TA feedback (formal equations were missing from the midterm)
  Section V (Classical ML methodology): full methodology with model families,
    strategies, hyperparameters, evaluation protocol
  Section V (Implementation Challenges): documented 2 challenges per TA feedback:
    (1) CPU DataLoader bottleneck (num_workers fix)
    (2) FastText OOV zero-vector collapse (std=0.1 fix)
  Section VI (Experiments and Results): classical results tables across 3 test sets,
    imbalance strategy comparison, confusion matrix analysis, error analysis,
    feature ablation table, threshold sensitivity table and figure,
    CPU inference time table, joint deep vs classical comparison table
  Section VII (Discussion): why tree >> linear (nonlinear URL feature separability);
    Why imbalance handling has minimal impact (mild 42/58 ratio);
    What FP/FN URL length distributions reveal about model failure modes;
    deployment context for threshold calibration
  Section IX (Team Contributions): updated bullet for Student C with all completed work
  GenAI Disclosure: updated disclosure section

- Built and completed all Student C presentation slides
