# AI Usage Log: Student C (Fatma Abu Al Wafa)
## Project 9: Neural Models for Phishing URL and Website Detection
## Queen's University | CISC 867 Deep Learning | 2026

---

## Tools Used
- Claude (Anthropic)
- ChatGPT (OpenAI)
---

## Allowed Uses — With Disclosure

The following uses comply with the course GenAI policy (brainstorming, debugging
assistance after the student attempts a fix first, and grammar/phrasing improvements.

### 1. Brainstorming and Concept Clarification

AI tools were used to:

- Brainstorm the design of the 3-strategy imbalance comparison (None / Weighted / SMOTE)
  and validate the decision to exclude SMOTE from tree-based models.
- Clarify the correct interpretation of `passthrough=False` in `StackingClassifier.`
  (meta-learner receives only base-model out-of-fold predictions, not original features).
- Clarify why `pos_weight = neg_count / pos_count` is the correct XGBoost imbalance
  correction formula.
- Clarify the difference between Gini importance and permutation importance, and why
  Gini importance may overestimate correlated features.
- Clarify the relationship between classification threshold and FPR/FNR trade-off for
  the deployment context analysis (corporate firewall vs. browser plugin).

### 2. Debugging Assistance (after the student's own attempt)

All bugs below were encountered, identified, and a student fix was attempted first.
AI assistance was used only to confirm or improve the fix.

**Bug 1 — `LinearSVC` has no `predict_proba()` method (Major)**

Student's observation: the unified evaluation function failed with
`AttributeError: 'LinearSVC' has no attribute 'predict_proba'`
when computing ROC-AUC.

Student's first attempt: catch the exception and skip ROC-AUC for SVM.

AI's suggestion confirmed: wrap `LinearSVC` in
`CalibratedClassifierCV(method='sigmoid', cv=3)` instead, which adds
calibrated probability output without skipping the metric.

Decision: The AI suggestion was accepted because it preserves the AUC comparison
across all models.

**Bug 2 — Validation set not being evaluated on the original (unscaled) test set (Minor)**

The student identified the best model inference in the error analysis section
used the wrong feature matrix (scaled instead of unscaled for RF/Stacking).

AI confirmed: add a flag `use_unscaled = any(tag in best_name for tag in ["RF", "XGB", "Stack"]).`
to select the correct feature matrix per model.

**Bug 3 — `f1_score` zero-division warning in ablation study (Minor)**

Student observed `UndefinedMetricWarning` when computing F1 on small feature subsets.

AI suggestion: add `zero_division=0` to `f1_score()` call.
Decision: accepted.

### 3. Code Quality Improvements

- Suggested using a `get_strategy()` helper function to map model names to their
  imbalance strategy label for the pivot-table visualisation, instead of
  duplicating logic across cells.
- Suggested passing `feature_cols` as an explicit parameter to the dataset class
  Rather than using a global variable.
- Reviewed the final deliverables checklist logic to ensure all 19 files are
  verified before the notebook is closed.

### 4. Grammar and Phrasing Improvements

AI tools were used for minor grammar and phrasing review of:

- Notebook markdown cells (section headers and interpretation paragraphs).
- Report sections after being written manually.
- These README and LOG.md files.

All content was written manually first. AI reviewed phrasing only — it did not
generate report sections or analytical conclusions.

---
## Not Allowed — Confirmed Not Used

In compliance with the course GenAI policy, AI tools were **not** used to:

- Generate experimental results or fake figures/tables.
- Write report sections, literature reviews, or methodology text from scratch.
- Implement the assigned project end-to-end or replace manual code execution.
- Make design decisions (model selection, hyperparameter choices) without testing.
- Generate code that was submitted without understanding it line by line.

All notebook cells were executed manually and verified before being committed
to the repository. All results in output CSVs and figures are based on
code that was run and checked by the student.
---

## Summary Table

| Area | AI Role | Policy Status |
|---|---|---|
| Imbalance strategy design | Brainstorming + validation | Allowed |
| `CalibratedClassifierCV` fix | Debugging (after student attempt) | Allowed |
| Gini importance interpretation | Concept clarification | Allowed |
| Threshold deployment guidance | Concept clarification | Allowed |
| Ablation zero-division fix | Debugging (minor) | Allowed |
| Feature matrix selection fix | Debugging (minor) | Allowed |
| Notebook markdown readability | Grammar / phrasing | Allowed |
| Report Discussion section | Grammar review of manual text | Allowed |
| Generating results / figures | — | Not used |
| Writing report sections | — | Not used |
| End-to-end implementation | — | Not used |
