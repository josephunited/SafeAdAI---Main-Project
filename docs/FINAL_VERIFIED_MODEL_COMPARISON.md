# SAFE-VISION — Authoritative Verified Model Comparison Report

**Audit Timestamp**: 2026-10-24T15:45:00+05:30  
**Git Branch**: `feature/phase-7b-dataset-v2-integration`  
**Dataset Evaluation Mode**: Development-set 5-fold cross-validation (*strictly no independent test set claims*)  
**Sample Count**: 194 Development Advertisements (134 Images, 60 Videos)  
**Authoritative Class Supports**: `SAFE_FOR_ALL`: 75 | `SAFE_14_PLUS`: 75 | `SAFE_18_PLUS`: 34 | `UNSAFE_FOR_ALL`: 10  
**Verification Result**: **100% MATHEMATICALLY VERIFIED & PASSED**

---

## 1. Root Cause Resolution of Historical Reporting Discrepancies

The following discrepancies identified in earlier project drafts have been audited directly against per-sample ground truth and predictions:

| Model | Reported Accuracy | Claimed Class TPs | True Verified Class TPs | Root Cause & Resolution |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Frozen Heuristic)** | 31.96% (62/194) | Sum = 67 | [48, 3, 9, 2] -> **Sum = 62** | **Resolved**: An uncorrected Phase 5B draft erroneously recorded SAFE_FOR_ALL recall as 53/75 (70.67%) instead of 48/75 (64.00%). Summing 53 + 3 + 9 + 2 produced 67. The verified matrix diagonal is `[48, 3, 9, 2]`, summing exactly to **62 correct** (31.96%). |
| **Candidate B (Enhanced Alcohol)** | 33.51% (65/194) | Sum = 69 | [48, 3, 12, 2] -> **Sum = 65** | **Resolved**: The same legacy draft recorded Candidate B SAFE_FOR_ALL as 52/75 (69.33%) instead of 48/75 (64.00%). Summing 52 + 3 + 12 + 2 produced 69. The verified matrix diagonal is `[48, 3, 12, 2]`, summing exactly to **65 correct** (33.51%). |
| **Logistic Regression (5-Fold CV)** | 38.14% (74/194) | Sum = 71 or 74 | [21, 34, 17, 2] -> **Sum = 74** | **Resolved**: In an exploratory unscaled run, the model achieved 71/194 (36.60%). The finalized, persisted 5-fold CV model (`StandardScaler(with_mean=False)` + Balanced class weights) achieved diagonal `[21, 34, 17, 2]`, summing exactly to **74 correct** (38.14%). 71 belonged to the unscaled run and is deprecated. |
| **Random Forest (5-Fold CV)** | 44.33% (86/194) | Sum = 86 | [33, 31, 18, 4] -> **Sum = 86** | **Consistent**: All diagonal cells `[33, 31, 18, 4]` sum exactly to **86 correct** (44.33%). |
| **Random Forest + Policy HITL** | 31.96% (62/194)* | Sum = 80 or 86 | [29, 14, 15, 4] -> **Sum = 62** | **Resolved**: 49 samples were abstained to human review. For the remaining 145 automated decisions, automated correct predictions sum to `[29, 14, 15, 4] = 62`. Earlier drafts erroneously added 18 abstained samples or treated human-review withholdings as correct 4-class predictions (e.g. 74+6=80). Per safety policy, **human-review withholding is never counted as an automated 4-class true positive**. Automated selective accuracy is 62/145 (**42.76%**). |

---

## 2. Authoritative Comparison Table

| Metric | Frozen Baseline (Heuristic) | Candidate B (Enhanced Alcohol) | Logistic Regression (5-Fold CV) | Random Forest (5-Fold CV) | Random Forest + HITL Routing (5-Fold CV) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fusion Architecture** | Hardcoded Rules | Gated Rules | L2 Balanced Linear | Bagged Decision Trees | Bagged Trees + Uncertainty Gate |
| **Total Development Samples** | 194 | 194 | 194 | 194 | 194 |
| **Abstained (Requires Human Review)** | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | **49 (25.26%)** |
| **Automated Decisions (Coverage)** | 194 (100.00%) | 194 (100.00%) | 194 (100.00%) | 194 (100.00%) | **145 (74.74%)** |
| **Automated Correct Predictions** | 62 / 194 | 65 / 194 | 74 / 194 | **86 / 194** | 62 / 145 |
| **Overall Raw 4-Class Accuracy** | 31.96% (62/194) | 33.51% (65/194) | 38.14% (74/194) | **44.33% (86/194)** | 31.96% (62/194) |
| **Selective Accuracy (On Automated)** | 31.96% (62/194) | 33.51% (65/194) | 38.14% (74/194) | 44.33% (86/194) | **42.76% (62/145)** |
| **Macro-Precision** | 0.2682 | 0.2825 | 0.3335 | **0.4006** | 0.3995 |
| **Macro-Recall** | 0.2862 | 0.3082 | 0.3583 | **0.4457** | 0.3536 (0.4612 auto) |
| **Macro-F1 Score** | 0.2317 | 0.2491 | 0.3356 | **0.4110** | 0.3478 (0.3997 auto) |
| **SAFE_FOR_ALL Recall** | 64.00% (48/75) | 64.00% (48/75) | 28.00% (21/75) | 44.00% (33/75) | 38.67% (29/75) |
| **SAFE_14_PLUS Recall** | 4.00% (3/75) | 4.00% (3/75) | **45.33% (34/75)** | 41.33% (31/75) | 18.67% (14/75) |
| **SAFE_18_PLUS Recall** | 26.47% (9/34) | 35.29% (12/34) | 50.00% (17/34) | **52.94% (18/34)** | 44.12% (15/34) |
| **UNSAFE_FOR_ALL Recall (Direct Rejection)** | 20.00% (2/10) | 20.00% (2/10) | 20.00% (2/10) | **40.00% (4/10)** | **40.00% (4/10)** |
| **Prohibited Withheld by Human Review** | 0 / 10 | 0 / 10 | 0 / 10 | 0 / 10 | **2 / 10 (20.00%)** |
| **Prohibited False Accept Rate (PFAR)** | 80.00% (8/10) | 80.00% (8/10) | 80.00% (8/10) | 60.00% (6/10) | **40.00% (4/10)** |
| **Adult-to-Underage Exposure (UGVR_18P)** | 58.82% (20/34) | 50.00% (17/34) | 38.24% (13/34) | 32.35% (11/34) | **17.65% (6/34)** |
| **Benign False Rejection Rate (FRR)** | 7.33% (11/150) | 7.33% (11/150) | 12.67% (19/150) | **6.00% (9/150)** | **6.00% (9/150)** |

---

## 3. Authoritative Confusion Matrices

### A. Baseline (Frozen Heuristic) — 4x4 Matrix
- Total Samples: 194 | Diagonal Sum: **62** | Accuracy: **31.96%**

```
True \ Pred        SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL  Total
SAFE_FOR_ALL                48             2            18               7     75
SAFE_14_PLUS                48             3            20               4     75
SAFE_18_PLUS                17             3             9               5     34
UNSAFE_FOR_ALL               5             0             3               2     10
Total                      118             8            50              18    194
```

### B. Candidate B (Enhanced Alcohol) — 4x4 Matrix
- Total Samples: 194 | Diagonal Sum: **65** | Accuracy: **33.51%**

```
True \ Pred        SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL  Total
SAFE_FOR_ALL                48             2            18               7     75
SAFE_14_PLUS                48             3            20               4     75
SAFE_18_PLUS                14             3            12               5     34
UNSAFE_FOR_ALL               5             0             3               2     10
Total                      115             8            53              18    194
```

### C. Logistic Regression (5-Fold CV) — 4x4 Matrix
- Total Samples: 194 | Diagonal Sum: **74** | Accuracy: **38.14%**

```
True \ Pred        SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL  Total
SAFE_FOR_ALL                21            28            14              12     75
SAFE_14_PLUS                24            34            10               7     75
SAFE_18_PLUS                 8             5            17               4     34
UNSAFE_FOR_ALL               2             3             3               2     10
Total                       55            70            44              25    194
```

### D. Random Forest (5-Fold CV) — 4x4 Matrix
- Total Samples: 194 | Diagonal Sum: **86** | Accuracy: **44.33%**

```
True \ Pred        SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL  Total
SAFE_FOR_ALL                33            23            17               2     75
SAFE_14_PLUS                25            31            12               7     75
SAFE_18_PLUS                 8             3            18               5     34
UNSAFE_FOR_ALL               3             1             2               4     10
Total                       69            58            49              18    194
```

### E. Random Forest + Policy-First HITL — 5-Column Matrix
- Total Samples: 194 | Automated Decisions: 145 | Abstentions: **49**
- Automated Diagonal Sum: **62** | Selective Accuracy: **42.76%** (62/145) | Raw Accuracy: **31.96%** (62/194)

```
True \ Pred        SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL  REQUIRES_HUMAN_REVIEW  Total
SAFE_FOR_ALL                29            11            15               2                     18     75
SAFE_14_PLUS                22            14            11               7                     21     75
SAFE_18_PLUS                 5             1            15               5                      8     34
UNSAFE_FOR_ALL               2             1             1               4                      2     10
Total                       58            27            42              18                     49    194
```

---

## 4. Safety Metrics & HITL Abstention Accounting

### A. Prohibited Content (UNSAFE_FOR_ALL, $N=10$)
- **Baseline**: 2 Rejected, 8 Leaked into Feed (**80.00% PFAR**).
- **Candidate B**: 2 Rejected, 8 Leaked into Feed (**80.00% PFAR**).
- **Logistic Regression**: 2 Rejected, 8 Leaked into Feed (**80.00% PFAR**).
- **Random Forest**: 4 Rejected, 6 Leaked into Feed (**60.00% PFAR**).
- **Random Forest + HITL**: 4 Rejected, 2 Withheld for Human Review, 4 Leaked into Feed (**40.00% PFAR**).  
  *Protective Gating = 60.00% (6/10 safe from unauthorized automatic publication).*

### B. Adult Content (SAFE_18_PLUS, $N=34$)
- **Baseline**: 9 Gated 18+, 20 Leaked to Underage (**58.82% UGVR**), 5 False UFA.
- **Candidate B**: 12 Gated 18+, 17 Leaked to Underage (**50.00% UGVR**), 5 False UFA.
- **Random Forest**: 18 Gated 18+, 11 Leaked to Underage (**32.35% UGVR**), 5 False UFA.
- **Random Forest + HITL**: 15 Gated 18+, 8 Withheld for Human Review, 6 Leaked to Underage (**17.65% UGVR**), 5 False UFA.  
  *Underage leakage drops by over 41 percentage points compared to baseline.*

### C. Benign Content Integrity (SAFE_FOR_ALL + SAFE_14_PLUS, $N=150$)
- **Baseline**: 11 Falsely Rejected (**7.33% FRR**).
- **Candidate B**: 11 Falsely Rejected (**7.33% FRR**).
- **Logistic Regression**: 19 Falsely Rejected (**12.67% FRR**).
- **Random Forest**: 9 Falsely Rejected (**6.00% FRR**).
- **Random Forest + HITL**: 9 Falsely Rejected (**6.00% FRR**), 39 Withheld for Human Review (26.00%).

---

## 5. Development Cross-Validation Limitations & Provenance

> [!CAUTION]
> **Strict Methodological Boundary**:  
> 1. All results for Logistic Regression, Random Forest, and Random Forest + HITL represent **5-fold Stratified Cross-Validation on the 194 development samples** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).  
> 2. These 194 samples were previously used during earlier phases for exploratory analysis and heuristic candidate design. Therefore, these performance metrics **CANNOT be claimed as independent, generalization test performance**.  
> 3. Zero held-out test set samples (the 88 quarantined candidates) were unblinded, evaluated, or accessed.  
> 4. Feature scaling (`StandardScaler`) was strictly fit on training folds and only transformed on validation folds to eliminate data leakage.

---

## 6. Selected Experimental Model Recommendation

- **Selected Experimental Candidate**: **Random Forest + Policy-First HITL (`learned_rf_hitl`)**.
- **Rationale**:
  1. Achieves the lowest Prohibited False Acceptance Rate (**40.00%**, cutting prohibited feed leaks in half compared to baseline).
  2. Reduces adult content exposure to teenagers from 58.82% down to **17.65%**.
  3. Achieves the lowest benign false rejection rate (**6.00%**, lower than baseline and candidate B).
  4. Preserves fail-safe human review routing for borderline cases rather than forcing uncalibrated high-risk automated classifications.

---

## 7. Exact Source Files Audited

1. [`results/phase8_learned_fusion/out_of_fold_predictions.csv`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/results/phase8_learned_fusion/out_of_fold_predictions.csv): Per-sample out-of-fold predictions and probabilities.
2. [`results/real_pipeline_baseline/saved_detector_outputs.json`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/results/real_pipeline_baseline/saved_detector_outputs.json): Authoritative baseline predictions on the 194 development benchmark.
3. [`results/phase8_learned_fusion/final_random_forest_estimator.joblib`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/results/phase8_learned_fusion/final_random_forest_estimator.joblib): Persisted 33-feature Random Forest model.
4. [`results/phase8_learned_fusion/confusion_matrices.csv`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/results/phase8_learned_fusion/confusion_matrices.csv): Authoritative 5-model confusion matrix breakdown.
5. [`results/phase8_learned_fusion/model_comparison.csv`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/results/phase8_learned_fusion/model_comparison.csv): Model summary metrics.
6. [`joseph_safead_core/backend2jb/models/experimental_detectors.py`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/joseph_safead_core/backend2jb/models/experimental_detectors.py): `LearnedRFFusionClassifier` implementation with dual fail-closed gates.
7. [`joseph_safead_core/backend2jb/ai/pipeline.py`](file:///c:/Users/hello/Downloads/Project%20main%20Instagram/joseph_safead_core/backend2jb/ai/pipeline.py): Pipeline execution with optional selectable flag `learned_rf_hitl`.
