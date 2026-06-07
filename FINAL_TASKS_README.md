# Project 9 — Final Submission Task List
**Team 9 | Queen's University**
**GitHub: https://github.com/zainbmaged/Deeplearning-URLCNN**
**Midterm Score: 98/100 — Target Final: 100/100**

---

## TA FEEDBACK — 3 MANDATORY FIXES FOR FINAL REPORT

These 3 fixes cost you 2 points in the midterm and must be done in the final report.

### FIX-1: Add GitHub link in the report text
In your final report, add this line in the Introduction or at the end of Section IV (Preprocessing):
> "All code, data splits, trained models, and reproduction scripts are available at:
> https://github.com/zainbmaged/Deeplearning-URLCNN"

---

### FIX-2: Write formal mathematical equations for ALL evaluation metrics
In Section V (Methodology) or in a dedicated "Evaluation Protocol" subsection, add these equations in LaTeX:

```latex
\begin{equation}
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}

\begin{equation}
\text{Precision} = \frac{TP}{TP + FP}
\end{equation}

\begin{equation}
\text{Recall} = \frac{TP}{TP + FN}
\end{equation}

\begin{equation}
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
    = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
\end{equation}

\begin{equation}
\text{ROC-AUC} = \int_0^1 \text{TPR}(t)\, d\,\text{FPR}(t)
\end{equation}

\begin{equation}
\text{FPR} = \frac{FP}{FP + TN}, \qquad
\text{FNR} = \frac{FN}{FN + TP}
\end{equation}

\begin{equation}
\text{Efficiency Score} = \frac{F_1}{\text{Params (millions)}}
\end{equation}
```

Where: TP = True Positives (phishing correctly detected), TN = True Negatives (legitimate correctly classified), FP = False Positives (legitimate wrongly flagged), FN = False Negatives (phishing missed).

---

### FIX-3: Document 2 technical implementation challenges in the report
Add a subsection called **"Implementation Challenges"** in Section V (Methodology) or Section VI (Discussion). Use exactly these two challenges — they are already real things your team solved:

**Challenge 1 — CPU DataLoader Bottleneck (Student B solved this):**
> "Initial training used num_workers=2 for parallel data loading. On CPU, this caused
> a bottleneck due to context-switching overhead: multiple workers competed for the
> single CPU, increasing batch loading time with no throughput gain. Setting
> num_workers=0 (single-process loading) reduced per-epoch time by approximately
> [X] minutes. This is a known PyTorch behavior on CPU-only machines."

**Challenge 2 — FastText Unknown Token Collapse (Student B solved this):**
> "FastText pretrained vectors cover English vocabulary. URL tokens such as
> 'paypa1', 'secure-login', and domain-specific substrings are out-of-vocabulary
> (OOV). Initializing all OOV embeddings as zero vectors caused all unknown tokens
> to be identical and unlearnable — the model could not differentiate between
> different phishing-pattern tokens that are all OOV. The fix was to initialize OOV
> embeddings with small random values (std=0.1), giving each token a unique
> starting point while keeping the scale comparable to pretrained vectors."

*(Fill in the actual time saving for Challenge 1 from your training logs.)*

---

## HOW TO USE THE REST OF THIS FILE

- `[DONE]` = confirmed in your notebooks or midterm
- `[TODO]` = missing, must be added
- `[PARTIAL]` = started but needs finishing

---

## STUDENT B — Zainb Zahran
*Notebook: `Deep_Model_v_2__.ipynb`*

### Done [DONE]
- [DONE] CharVocab (size 92, min_freq=3, cap=100)
- [DONE] URLDataset with 10% token-masking augmentation + WeightedRandomSampler
- [DONE] Multi-Scale Char-CNN: embed(64) → Conv1D {k=3,5,7, 128 filters} → BN+ReLU+GlobalMaxPool → Dropout(0.3) → FC(384→128→1) → 179,329 params
- [DONE] BCEWithLogitsLoss pos_weight=1.368, Adam lr=1e-3 wd=1e-4, ReduceLROnPlateau factor=0.5 patience=3, grad clip 1.0, early stopping patience=5
- [DONE] Char-CNN trained 15 epochs: val acc=99.12%, F1=0.9896, efficiency=5.52 F1/M-params
- [DONE] CNN+BiLSTM implemented and trained (checkpoint: cnn_bilstm_best.pth)
- [DONE] FastText+CNN implemented and trained (checkpoint: cnn_fasttext_(1).pth)
- [DONE] FastText OOV smoothing fix (std=0.1 random init for unknown tokens)
- [DONE] num_workers=0 CPU fix
- [DONE] Efficiency: F1 per million TRAINED params (excludes pretrained embedding for FastText)
- [DONE] Validation comparison table: Char-CNN vs CNN+BiLSTM vs FastText+CNN

### Missing [TODO]

#### TODO-B1: Hybrid Feature Fusion Model [MOST IMPORTANT — 20 pts Originality]
Your main novel contribution. Promised in midterm. Not yet in notebook.

Architecture: Char-CNN branch (same as before) + 23 handcrafted features branch (FC: 23→32) → concatenate (384+32=416) → Dropout → FC(416→128) → FC(128→1).

```python
class HybridFusionCNN(nn.Module):
    def __init__(self, vocab_size, num_handcrafted=23, embed_dim=64,
                 num_filters=128, kernel_sizes=[3,5,7], dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        self.feat_fc = nn.Linear(num_handcrafted, 32)
        self.dropout = nn.Dropout(dropout)
        conv_out = num_filters * len(kernel_sizes)  # 384
        self.fc1 = nn.Linear(conv_out + 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_char, x_feat):
        x = self.embedding(x_char).permute(0, 2, 1)
        conv_outs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            c = F.relu(bn(conv(x)))
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_outs.append(c)
        cnn_out  = torch.cat(conv_outs, dim=1)      # [B, 384]
        feat_out = F.relu(self.feat_fc(x_feat))     # [B, 32]
        fused    = torch.cat([cnn_out, feat_out], dim=1)  # [B, 416]
        fused    = self.dropout(fused)
        out      = F.relu(self.fc1(fused))
        out      = self.dropout(out)
        return self.fc2(out).squeeze(1)
```

Dataset class for Hybrid (loads URL chars AND handcrafted features):
```python
class HybridDataset(Dataset):
    def __init__(self, df, feat_df, vocab, max_len=200, augment=False, scaler=None):
        self.urls   = df['URL'].values
        self.labels = df['Label'].values
        self.vocab  = vocab
        self.max_len = max_len
        self.augment = augment
        feat_cols = [c for c in feat_df.columns if c != 'Label']
        feats = feat_df[feat_cols].values.astype(np.float32)
        if scaler is not None:
            feats = scaler.transform(feats)
        self.feats = feats

    def __len__(self): return len(self.urls)

    def __getitem__(self, idx):
        encoded = self.vocab.encode(self.urls[idx], self.max_len, pad_to_max=True)
        if self.augment and np.random.random() < 0.1:
            mask = np.random.random(len(encoded)) < 0.05
            for i in range(len(encoded)):
                if mask[i] and encoded[i] > 1:
                    encoded[i] = self.vocab.unkown_idx
        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(self.feats[idx], dtype=torch.float),
                torch.tensor(self.labels[idx], dtype=torch.float))
```

Load feature files from Student A and train:
```python
import joblib
train_feat = pd.read_csv('../data/features/train_features.csv')
val_feat   = pd.read_csv('../data/features/val_features.csv')
test_feat  = pd.read_csv('../data/features/test_features.csv')
scaler     = joblib.load('../models/standard_scaler.joblib')

train_hyb = HybridDataset(train_df, train_feat, vocab, 200, augment=True,  scaler=scaler)
val_hyb   = HybridDataset(val_df,   val_feat,   vocab, 200, augment=False, scaler=scaler)

# Train with same config: BCEWithLogitsLoss pos_weight=1.368, Adam, scheduler, early stopping
# Save checkpoint as: hybrid_fusion_best.pth
```

---

#### TODO-B2: Evaluate ALL 4 Deep Models on Test Set
Right now all results are validation only. Final report needs test-set numbers.

```python
test_df      = pd.read_csv('test.csv')
test_dataset = URLDataset(test_df, vocab, MAX_LEN, augment=False)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

# Load each best checkpoint and run:
test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
metric_test = compute_metrics(test_labels, test_preds)
print("TEST:", metric_test)
```

Do this for: Char-CNN, CNN+BiLSTM, FastText+CNN, Hybrid Fusion.
Also evaluate on test_balanced.csv and test_imbalanced_80_20.csv (from Student A).

---

#### TODO-B3: Inference Time Benchmark
Needed to prove "CPU-deployable" claim.

```python
import time
model.eval()
sample_ds = URLDataset(test_df.head(1000), vocab, MAX_LEN, augment=False)
sample_ldr = DataLoader(sample_ds, batch_size=256, shuffle=False, num_workers=0)
t0 = time.time()
with torch.no_grad():
    for x, y in sample_ldr:
        _ = model(x.to(device))
ms_per_url = (time.time() - t0) / 1000 * 1000
print(f"{ms_per_url:.3f} ms per URL")
```

Do for all 4 models. Add column "ms/URL" to final comparison table.

---

#### TODO-B4: Final 4-Model Comparison Table
Build this table and save as reports/tables/deep_model_comparison_final.csv:

| Model | Params(M) | TrainedParams(M) | ValF1 | TestF1 | TestAUC | ms/URL | F1/M-trained |
|---|---|---|---|---|---|---|---|
| Char-CNN | 0.179 | 0.179 | 0.9896 | ? | ? | ? | 5.52 |
| CNN+BiLSTM | ? | ? | ? | ? | ? | ? | ? |
| FastText+CNN | ? | ? | ? | ? | ? | ? | ? |
| Hybrid Fusion | ? | ? | ? | ? | ? | ? | ? |

---

#### TODO-B5: Training Curves Plot
Plot loss and accuracy curves per epoch for all 4 models (2×2 or 4×2 subplots).
Save as: reports/figures/deep_model_training_curves.png

---

#### TODO-B6: Confusion Matrix on Test Set (Best Deep Model)
3-subplot figure (original / balanced / imbalanced test sets), same style as Student C.
Save as: reports/figures/deep_model_confusion_matrices.png

---

## STUDENT C — Fatma Abu Al Wafa
*Notebook: `Student_C_baseline_models.ipynb`*

### Done [DONE]
- [DONE] LR, SVM, RF, XGBoost — 3 imbalance strategies each (none / weighted / SMOTE)
- [DONE] Stacking Ensemble: RF+XGB+LR base learners, LR meta-learner, cv=5, passthrough=False
- [DONE] Unified evaluate() + evaluate_all_sets() on 4 splits (val / original / balanced / imbalanced)
- [DONE] RF Gini feature importance (top-5: path_length, hostname_length, shannon_entropy, path_depth, url_length)
- [DONE] ROC curves for all 11 models on original test set (figure saved)
- [DONE] Confusion matrices: Stacking Ensemble on 3 test sets (TN=10909, FP=337, FN=448, TP=7773)
- [DONE] Error analysis: FP/FN URL length histograms + sample URLs saved
- [DONE] CPU training times: LR≈4.9s, SVM≈3.7s, RF≈44.4s, XGB≈4.2s, Stacking=316.2s
- [DONE] Best model saved: models/best_baseline_model.joblib
- [DONE] Deliverables checklist cell (34 output files verified)

### Missing [TODO]

#### TODO-C1: Feature Subset Ablation Study
Promised in midterm. Required for Originality marks.

```python
from sklearn.metrics import f1_score

# fi = your feature importance dataframe (already computed in Section 11)
feature_subsets = {
    'Top-5':  ['path_length','hostname_length','shannon_entropy','path_depth','url_length'],
    'Top-10': fi['feature'].head(10).tolist(),
    'All-23': FEATURE_COLS,
}

ablation_rows = []
for name, cols in feature_subsets.items():
    idx = [FEATURE_COLS.index(c) for c in cols]
    rf_abl = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    n_jobs=-1, random_state=RANDOM_STATE)
    rf_abl.fit(X_tr_u[:, idx], y_tr)
    val_f1  = f1_score(y_val, rf_abl.predict(X_val_u[:, idx]))
    test_f1 = f1_score(y_te,  rf_abl.predict(X_te_u[:, idx]))
    ablation_rows.append({'Subset': name, 'N_features': len(cols),
                          'Val_F1': round(val_f1,4), 'Test_F1': round(test_f1,4)})
    print(f"{name:8s}: Val F1={val_f1:.4f}  Test F1={test_f1:.4f}")

abl_df = pd.DataFrame(ablation_rows)
abl_df.to_csv(report_table_dir / 'feature_ablation_results.csv', index=False)
```

---

#### TODO-C2: Threshold Sensitivity Analysis
Promised in midterm. Shows security deployment tradeoff (FNR vs Precision).

```python
y_prob_best = get_probabilities(best_model, X_te_u)
rows = []
for t in np.arange(0.1, 0.95, 0.05):
    yp = (y_prob_best >= t).astype(int)
    tp = ((yp==1)&(y_te==1)).sum(); fp = ((yp==1)&(y_te==0)).sum()
    fn = ((yp==0)&(y_te==1)).sum(); tn = ((yp==0)&(y_te==0)).sum()
    rows.append({'threshold': round(t,2),
        'precision': round(tp/(tp+fp+1e-9),4),
        'recall':    round(tp/(tp+fn+1e-9),4),
        'f1':        round(2*tp/(2*tp+fp+fn+1e-9),4),
        'fpr':       round(fp/(fp+tn+1e-9),4),
        'fnr':       round(fn/(fn+tp+1e-9),4)})

thresh_df = pd.DataFrame(rows)
thresh_df.to_csv(report_table_dir / 'threshold_sensitivity.csv', index=False)

fig, ax = plt.subplots(figsize=(9,5))
for col, marker in [('precision','o'),('recall','s'),('f1','^'),('fnr','x')]:
    ax.plot(thresh_df['threshold'], thresh_df[col], label=col.capitalize(), marker=marker)
ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
ax.set_title('Threshold Sensitivity — Stacking Ensemble (Original Test Set)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figure_dir / 'threshold_sensitivity.png', dpi=300)
plt.show()
```

---

#### TODO-C3: CPU Inference Time for Classical Models

```python
inference_rows = []
for name, model, X_s in [
    ('LR (no handling)',  lr_none,    X_te_s[:1000]),
    ('RF (no handling)',  rf_none,    X_te_u[:1000]),
    ('XGB (no handling)', xgb_none,   X_te_u[:1000]),
    ('Stacking Ensemble', stack_model, X_te_u[:1000]),
]:
    t0 = time.time()
    _ = model.predict(X_s)
    ms = (time.time()-t0)/1000*1000
    inference_rows.append({'model': name, 'ms_per_url': round(ms,4)})
    print(f"{name}: {ms:.4f} ms/URL")

pd.DataFrame(inference_rows).to_csv(
    report_table_dir/'classical_inference_time.csv', index=False)
```

---

#### TODO-C4: Joint Deep + Classical Final Comparison Table
After Student B finishes, merge both results into one table for the final report.

```python
# After Student B sends deep_model_comparison_final.csv:
deep = pd.read_csv(report_table_dir / 'deep_model_comparison_final.csv')
# Add columns: Model, Type='Deep', TestF1, TestAUC, Params(M), ms/URL

# Classical best results (already in results_df):
classical_best = results_df[
    (results_df['dataset']=='test_original') &
    (results_df['model'].isin(['Stacking Ensemble','RF (no handling)','XGB (weighted)']))
][['model','f1','auc_roc']].rename(columns={'f1':'TestF1','auc_roc':'TestAUC'})
classical_best['Type'] = 'Classical'

# Combine and save
joint = pd.concat([classical_best, deep], ignore_index=True)
joint.to_csv(report_table_dir / 'final_all_models_comparison.csv', index=False)
```

---

## STUDENT A — Ahmed Mohamed Al-Shobaki
*Notebook: `01_studentA_data_preparation.ipynb`*

### Done [DONE]
- [DONE] Audit: 149,726 rows, no missing/empty URLs, no conflicting labels, valid binary labels
- [DONE] 19,950 duplicates removed → 129,776 clean URLs (74,972 legit / 54,804 phishing)
- [DONE] Stratified 70/15/15 split (RANDOM_STATE=42): 90,843 / 19,466 / 19,467
- [DONE] 23 features: URL/hostname/path/query lengths, dot/hyphen/digit/slash counts, digit_ratio, shannon_entropy, has_https, has_ip_address, subdomain_depth, path_depth, hostname_has_hyphen, tld_in_path, suspicious_keyword_count
- [DONE] StandardScaler fitted on train only → models/standard_scaler.joblib
- [DONE] All scaled + unscaled feature files saved
- [DONE] Balanced test (50/50, n=16,442) + Imbalanced test (80/20, n=14,057)
- [DONE] SMOTE training file (104,960 samples, 50/50)
- [DONE] EDA: class distribution, URL length boxplot/histogram, binary feature rates, TLD analysis, correlation heatmap
- [DONE] 34 output files with automated assertions

### Missing [TODO]

#### TODO-A1: Confirm raw URL split files exist for Student B
Student B's deep model notebook loads `train.csv`, `val.csv`, `test.csv` (raw URL+Label, not features).
Student C's error analysis also loads `feature_dir / "test.csv"`.
Check your data folder. If these raw split files do not exist, add:

```python
# Add at end of Section 2 (after creating splits):
train_df[['URL','Label']].to_csv(split_dir / 'train.csv', index=False)
val_df[['URL','Label']].to_csv(split_dir   / 'val.csv',   index=False)
test_df[['URL','Label']].to_csv(split_dir  / 'test.csv',  index=False)
# Also save balanced and imbalanced test URL files:
# (you need to rebuild test_bal and test_imb URL versions from the split indices)
print("Raw URL split files saved.")
```

Share the full data folder path on Google Drive with Student B immediately.

---

#### TODO-A2: Verify EDA Figures are Saved Correctly
Check reports/figures/ contains these files (needed in final report):
- class_distribution.png — shows 74,972 / 54,804 after cleaning
- url_length_boxplot.png — capped at 1000, upper whisker ≈ 200
- binary_feature_rates.png — hostname_has_hyphen bar clearly visible
- tld_analysis.png
- feature_correlation_heatmap.png

If any are missing, re-run the relevant EDA cell and save explicitly with:
```python
plt.savefig(figure_dir / 'filename.png', dpi=300, bbox_inches='tight')
```

---

## TEAM-WIDE TASKS

### TODO-T1: LOG.md in GitHub [REQUIRED BY COURSE POLICY]
The course GenAI policy requires LOG.md with weekly entries matching Git commit history.
Create at repo root. Template:

```markdown
# Development Log — Team 9, Project 9

## Week 1
### Ahmed (Student A)
- Downloaded Mendeley 2026 dataset (DOI: 10.17632/3jddhy2f6s.1)
- Audited 149,726 rows: no missing values, no conflicting labels found
- Removed 19,950 duplicate URLs → 129,776 clean records
- Created stratified 70/15/15 splits, RANDOM_STATE=42
- Decision: used parsed.hostname (not netloc) to avoid port number contamination in hostname features

## Week 2
### Ahmed (Student A)
- Engineered 23 lexical URL features
- Fitted StandardScaler on train only; saved as standard_scaler.joblib
- Created balanced (50/50) and imbalanced (80/20) evaluation test sets
- Created SMOTE-balanced training file (104,960 samples)

### Fatma (Student C)
- Implemented LR, SVM, RF, XGBoost baselines with 3 imbalance strategies each
- Key finding: tree-based models achieve AUC≈0.99 vs linear models AUC≈0.91
- Challenge solved: CalibratedClassifierCV needed for LinearSVC probability estimates

## Week 3
### Zainb (Student B)
- Built CharVocab and URLDataset classes with augmentation
- Trained Multi-Scale Char-CNN: val acc=99.12%, F1=0.9896 after 15 epochs
- Challenge solved: num_workers=2 caused CPU bottleneck → switched to num_workers=0
- Implemented CNN+BiLSTM; training interrupted, resumed from checkpoint
- Challenge solved: FastText OOV tokens initialized as zeros → all identical → fixed with std=0.1 random init
- Implemented FastText+CNN; trained 15 epochs

### Fatma (Student C)
- Built Stacking Ensemble (RF+XGB+LR base, LR meta, cv=5)
- Stacking achieves best F1=0.9519, AUC=0.9918 on original test set
- RF feature importance: top-5 features account for >50% of importance

## Week 4
### [Fill in your actual Week 4 work here before submitting]
```

---

### TODO-T2: README.md in GitHub [REQUIRED]
Replace or create README.md at repo root:

```markdown
# Phishing URL Detection — Team 9

Multi-Scale Character-Level CNN and Hybrid Feature Fusion for Lightweight Phishing URL Detection

## Team
| Student | Role |
|---|---|
| Ahmed Mohamed Al-Shobaki (25LSWJ@queensu.ca) | Data Pipeline (Student A) |
| Zainb Zahran (25vpgv@queensu.ca) | Deep Learning Models (Student B) |
| Fatma Abu Al Wafa (25qgkp@queensu.ca) | Classical ML Baselines (Student C) |

## Dataset
Mendeley 2026 Phishing URL Dataset — DOI: 10.17632/3jddhy2f6s.1
129,776 URLs (74,972 legitimate / 54,804 phishing) after cleaning

## Repository Structure
```
├── notebooks/
│   ├── 01_studentA_data_preparation.ipynb
│   ├── Deep_Model_v_2__.ipynb
│   └── Student_C_baseline_models.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   ├── features/
│   └── features_scaled/
├── models/
│   ├── standard_scaler.joblib
│   ├── best_baseline_model.joblib
│   ├── char_cnn_best.pth
│   ├── cnn_bilstm_best.pth
│   ├── cnn_fasttext_(1).pth
│   └── hybrid_fusion_best.pth
├── reports/
│   ├── figures/
│   └── tables/
├── LOG.md
├── README.md
└── requirements.txt
```

## How to Reproduce

### Step 1 — Data Preparation
```bash
jupyter nbconvert --to notebook --execute notebooks/01_studentA_data_preparation.ipynb
```

### Step 2 — Classical Baselines
```bash
jupyter nbconvert --to notebook --execute notebooks/Student_C_baseline_models.ipynb
```

### Step 3 — Deep Models
```bash
jupyter nbconvert --to notebook --execute notebooks/Deep_Model_v_2__.ipynb
```

## Requirements
See requirements.txt. Main packages:
torch==2.3.0+cpu, torchtext==0.18.0, scikit-learn, xgboost, imbalanced-learn, pandas, numpy

## Key Results
| Model | Test F1 | Test AUC | Params |
|---|---|---|---|
| Stacking Ensemble | 0.9519 | 0.9918 | — |
| Char-CNN | 0.9896* | — | 179K |
| CNN+BiLSTM | TBD | TBD | TBD |
| FastText+CNN | TBD | TBD | TBD |
| Hybrid Fusion | TBD | TBD | TBD |

*validation (test evaluation in progress)

## Reproducibility
All experiments use RANDOM_STATE=42 and torch.manual_seed(42) and numpy.random.seed(42).
```

---

### TODO-T3: requirements.txt in GitHub

```
torch==2.3.0+cpu
torchtext==0.18.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
tqdm>=4.65.0
```

---

### TODO-T4: Final Report Section Assignments

| Section | Writer | Notes |
|---|---|---|
| Title + Abstract | All 3 | Update: mention all 4 deep models + final numbers |
| 1. Introduction | Student C | Keep midterm; update contribution statement; ADD GitHub link |
| 2. Related Work | Student B | Keep midterm; add 1–2 sentences on Hybrid Fusion motivation |
| 3. Dataset | Student A | Keep midterm Section III almost unchanged |
| 4. Preprocessing | Student A | Keep midterm Section IV; add final file counts |
| 5. Methodology — Eval metrics | Student C | ADD all 7 LaTeX equations (see FIX-2 above) |
| 5. Methodology — Classical | Student C | Keep midterm; add ablation + threshold descriptions |
| 5. Methodology — Deep Models | Student B | Keep midterm; add Hybrid Fusion architecture + diagram |
| 5. Implementation Challenges | Student B | ADD 2 challenges (see FIX-3 above) |
| 6. Experiments + Results | All 3 | NEW: final tables all models on test set, curves, confusion matrices |
| 7. Discussion | Student C | NEW: Why Hybrid > Char-CNN? Why tree >> linear? Error overlap analysis |
| 8. Limitations + Future Work | Student B | NEW: CPU limits, max_len truncation, no real-time demo |
| 9. Conclusion | All 3 | 1 paragraph with best numbers |
| Team Contributions | All 3 | Update from midterm |
| GenAI Disclosure | All 3 | Keep midterm text; update if anything changed |
| References | All 3 | Keep all 20 midterm refs; add any new ones |

---

### TODO-T5: Presentation (20 min + Q&A)

| # | Slide Content | Speaker | Minutes |
|---|---|---|---|
| 1 | Title, team names, GitHub link | — | — |
| 2 | Motivation: 1.9M phishing attacks, 51% subdomain growth | Student C | 1.5 |
| 3 | Problem + scope: raw URL → binary classification, CPU-only | Student C | 1 |
| 4 | Dataset: 129,776 URLs, cleaning pipeline (duplicates, leakage prevention) | Student A | 1.5 |
| 5 | 23 features + top-5 importance bar chart (path_length dominates) | Student A | 1.5 |
| 6 | Classical results table: tree >> linear, Stacking F1=0.9519 AUC=0.9918 | Student C | 2 |
| 7 | ROC curve figure + imbalance strategy comparison | Student C | 1 |
| 8 | 4 deep model architectures diagram | Student B | 2 |
| 9 | Training curves + validation results | Student B | 1.5 |
| 10 | Final comparison table: all models, test F1, AUC, params, ms/URL | Student B | 2 |
| 11 | Hybrid Fusion: why it works (feature importance motivates fusion) | Student B | 1 |
| 12 | Threshold sensitivity + error analysis | Student C | 1 |
| 13 | Limitations + Future Work | Student A | 1 |
| 14 | Conclusion: top 3 findings with numbers | All | 0.5 |

**Q&A preparation — each member must be ready to explain:**
- Student A: why parsed.hostname not netloc, how SMOTE file was created, why scaler fitted on train only
- Student B: why 3 kernel sizes, why global max pooling, how BiLSTM takes CNN output, why FastText OOV fix was needed
- Student C: what scale_pos_weight means in XGBoost, why passthrough=False in stacking, what CalibratedClassifierCV does

---

## FINAL GRADING CHECKLIST

### Technical Depth & Correctness (30 pts)
- [ ] All 4 deep models evaluated on test set
- [ ] Classical baselines on all 3 test distributions (already done)
- [ ] Hybrid Fusion implemented and results included
- [ ] No data leakage (scaler fitted on train only) ✓
- [ ] RANDOM_STATE=42 everywhere ✓

### Originality & Insight (20 pts)
- [ ] Hybrid Feature Fusion model built and compared (TODO-B1)
- [ ] Feature ablation study top-5/top-10/all-23 (TODO-C1)
- [ ] Threshold sensitivity with deployment insight (TODO-C2)
- [ ] Implementation challenges documented (FIX-3)

### Implementation & Reproducibility (20 pts)
- [ ] README.md in GitHub (TODO-T2)
- [ ] LOG.md in GitHub (TODO-T1)
- [ ] requirements.txt (TODO-T3)
- [ ] GitHub link in report text (FIX-1)
- [ ] All notebooks run clean top-to-bottom
- [ ] 34 output files verified ✓

### Written Reports (20 pts)
- [ ] Midterm submitted ✓ (98/100)
- [ ] Final report: all sections written
- [ ] LaTeX metric equations added (FIX-2)
- [ ] GitHub link in report (FIX-1)
- [ ] 2 implementation challenges documented (FIX-3)
- [ ] GenAI disclosure section ✓

### Presentation & Teamwork (10 pts)
- [ ] 14 slides built (TODO-T5)
- [ ] All 3 members speak
- [ ] Fits in 20 minutes
- [ ] Each member can explain their own code
- [ ] Workload breakdown clear in report ✓

---

## DO IN THIS ORDER

1. Student A → TODO-A1 (confirm raw CSV files, share Google Drive with Student B)
2. Student B → TODO-B1 (Hybrid Fusion — this takes longest)
3. Student B → TODO-B2 (test-set evaluation all 4 models)
4. Student C → TODO-C1 + C2 (ablation + threshold — both fast)
5. Student B → TODO-B3 + B4 + B5 + B6 (inference time, tables, curves, confusion matrices)
6. Student C → TODO-C3 + C4 (inference time, joint table)
7. All → TODO-T1 + T2 + T3 + FIX-1 (GitHub files)
8. All → TODO-T4 (write final report — Student B writes Hybrid Fusion section first)
9. All → TODO-T5 (presentation slides)
