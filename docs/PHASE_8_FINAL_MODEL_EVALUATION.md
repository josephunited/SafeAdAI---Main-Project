# SAFE-VISION — Phase 8: Final Model Evaluation Report

**Project**: SAFE-VISION (SafeAd AI Multimodal Advertisement Moderation System)  
**Document**: Final Multimodal Model Evaluation, Detector Comparisons & Safety Metrics  
**Date**: October 22, 2026  
**Status**: COMPLETE (Verified on 194 Clean Development Samples; Held-Out Test Set Isolated)  
**Recommendation**: **NO-GO FOR AUTOMATIC PRODUCTION PUBLICATION** (Candidate B approved as experimental baseline; human verification required for publication).

---

## 1. Executive Evaluation Summary

Phase 8 concludes the empirical model evaluation of the SAFE-VISION advertising safety pipeline across the **194 eligible clean development samples**.

```text
========================================================================================
MODEL PERFORMANCE COMPARISON (Clean Development Cohort N=194)
----------------------------------------------------------------------------------------
Configuration          Overall Acc   Macro-F1   18+ Recall   PFAR (Prohibited Released)
----------------------------------------------------------------------------------------
Production Baseline       31.96%      0.2317      26.47%              80.00% (8/10)
Candidate B (Alcohol)     33.51%      0.2491      35.29%              80.00% (8/10)
OOF Calibrated            34.02%      0.2575      29.41%              80.00% (8/10)
Hybrid Safety + HITL      24.23%*     0.2020      29.41%              70.00% (7/10)
----------------------------------------------------------------------------------------
* Hybrid Safety achieves 77.84% automated coverage, 31.13% auto accuracy, and slashes
  underage gating violations by 47% while routing 43 ambiguous ads to Human Review.
========================================================================================
```

### Key Academic Findings:
1. **Candidate B Verified**: Candidate B (Baseline + `EnhancedAlcoholDetector`) improves development accuracy from **62/194 (31.96%)** to **65/194 (33.51%)**, and increases `SAFE_18_PLUS` recall from **26.47%** to **35.29%** (+8.82%).
2. **Three Changed Samples**: Exactly 3 samples changed prediction between Baseline and Candidate B: `SAFEAD_S18P_IMG_002`, `SAFEAD_S18P_IMG_006`, and `SAFEAD_S18P_IMG_031`. All three moved from `SAFE_FOR_ALL` to `SAFE_18_PLUS`, correctly matching ground truth.
3. **Statistical Superiority Not Claimed**: A 3-sample gain ($+1.55\%$) across 194 instances does **not** prove statistical superiority ($p > 0.10$). However, Candidate B provides superior regulatory compliance by correctly detecting branded beer and spirits without regressing on non-alcoholic controls (soda, olive oil, 0.0% malt beverages).
4. **Prohibited False Acceptance Severity**: Both Baseline and Candidate B exhibit an **80.00% Prohibited False Acceptance Rate (PFAR)** (8 out of 10 prohibited ads released). The underlying neural detectors fail to detect subtle financial scams and visual hazards without human oversight.
5. **Cohort Isolation Maintained**: 100% of evaluated data originates from the approved 194 development samples. Zero historical held-out test samples ($N=88$) and zero synthetic cards ($N=118$) were evaluated.

---

## 2. Cohort Provenance and Assertion Verification

Before evaluation execution, the Phase 8 engine asserted cohort integrity:
- **Total Catalog Size**: 414 assets.
- **Approved Development Cohort**: Exactly 194 samples (`results/real_pipeline_baseline/saved_detector_outputs.json`).
- **Historical Held-Out Partition Candidates**: 88 samples — **Zero Overlap Confirmed**.
- **Synthetic OCR Cards**: 118 samples — **Zero Overlap Confirmed**.
- **New Scam Supplementary Specimens**: 14 samples — **Zero Overlap Confirmed**.
- **Source Label Version**: Original frozen development labels preserved.
- **Commit SHA**: Main application `9b532a66f5918d9ec5c8f4636ae0c9cf22a56f97`; Dataset `0e8883856cad683dbeaf2c153660df25fa8c4c11`.

---

## 3. Four-Class Performance Breakdown

### 3.1 Confusion Matrices

#### Baseline Confusion Matrix ($N=194$)
```text
                      PREDICTED CLASS
ACTUAL CLASS       SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL   TOTAL
SAFE_FOR_ALL            48             2            18               7          75
SAFE_14_PLUS            48             3            20               4          75
SAFE_18_PLUS            17             3             9               5          34
UNSAFE_FOR_ALL           5             0             3               2          10
TOTAL PREDICTED        118             8            50              18         194
```

#### Candidate B Confusion Matrix ($N=194$)
```text
                      PREDICTED CLASS
ACTUAL CLASS       SAFE_FOR_ALL  SAFE_14_PLUS  SAFE_18_PLUS  UNSAFE_FOR_ALL   TOTAL
SAFE_FOR_ALL            48             2            18               7          75
SAFE_14_PLUS            48             3            20               4          75
SAFE_18_PLUS            14             3            12               5          34
UNSAFE_FOR_ALL           5             0             3               2          10
TOTAL PREDICTED        115             8            53              18         194
```

### 3.2 Per-Class Precision, Recall, and F1 Metrics

| Category | Support | Baseline Precision | Baseline Recall | Baseline F1 | Candidate B Precision | Candidate B Recall | Candidate B F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `SAFE_FOR_ALL` | 75 | 40.68% | 64.00% | 0.4974 | 41.74% | 64.00% | 0.5053 |
| `SAFE_14_PLUS` | 75 | 37.50% | 4.00% | 0.0723 | 37.50% | 4.00% | 0.0723 |
| `SAFE_18_PLUS` | 34 | 18.00% | 26.47% | 0.2143 | 22.64% | 35.29% | 0.2759 |
| `UNSAFE_FOR_ALL`| 10 | 11.11% | 20.00% | 0.1429 | 11.11% | 20.00% | 0.1429 |
| **Macro Average** | **194** | **26.82%** | **28.62%** | **0.2317** | **28.25%** | **30.82%** | **0.2491** |

---

## 4. Safety-Critical Failure Analysis

### 4.1 Prohibited False Acceptance Failure (PFAR = 80.00%)
Out of 10 ground-truth prohibited advertisements in the development set, only **2 were correctly rejected**:
- `SAFEAD_UFA_VID_001` (Correctly rejected: kinetic violence score 0.90, `visual_unsafe=True`).
- `SAFEAD_UFA_VID_003` (Correctly rejected: kinetic violence score 0.90, `visual_unsafe=True`).

The remaining **8 prohibited samples were falsely accepted**:
1. `SAFEAD_UFA_IMG_021`: Classified as `SAFE_18_PLUS` ($S_{\text{risk}} = 28.00$).
2. `SAFEAD_UFA_IMG_026`: Classified as `SAFE_FOR_ALL` ($S_{\text{risk}} = 0.40$).
3. `SAFEAD_UFA_IMG_027`: Classified as `SAFE_FOR_ALL` ($S_{\text{risk}} = 13.15$).
4. `SAFEAD_UFA_IMG_034`: Classified as `SAFE_FOR_ALL` ($S_{\text{risk}} = 0.40$).
5. `SAFEAD_UFA_IMG_035`: Classified as `SAFE_FOR_ALL` ($S_{\text{risk}} = 1.00$).
6. `SAFEAD_UFA_IMG_054`: Classified as `SAFE_FOR_ALL` ($S_{\text{risk}} = 0.40$).
7. `SAFEAD_UFA_IMG_068`: Classified as `SAFE_18_PLUS` ($S_{\text{risk}} = 51.04$).
8. `SAFEAD_UFA_VID_002`: Classified as `SAFE_18_PLUS` ($S_{\text{risk}} = 28.00$).

**Root Cause**: The visual and text safety detectors produced near-zero threat scores ($0.00$ to $0.05$) on financial phishing schemes and subtle deceptive ads. The models lack fine-tuned advertising policy understanding.

### 4.2 Underage Gating Violations (18+ Exposed to Minors)
- **Baseline**: 20 out of 34 adult ads (58.82%) were misclassified as suitable for general/teen audiences (17 as SFA, 3 as S14P).
- **Candidate B**: 17 out of 34 adult ads (50.00%) were misclassified as suitable for general/teen audiences (14 as SFA, 3 as S14P).
- **Candidate B Impact**: Slashed underage leaks by 3 samples (+8.82% improvement) due to alcohol brand recognition.

---

## 5. Exploratory Sensitivity Analysis on AI-Proposed Label Corrections

> [!CAUTION]
> **Exploratory Sensitivity Analysis Only — NOT Independently Validated Accuracy**:
> The dataset developer proposed correcting 70 labels in the dataset, affecting **53 samples** within our 194 development cohort (primarily reclassifying adolescent streetwear and routine electronics from `SAFE_14_PLUS` to `SAFE_FOR_ALL`). These corrections have **not** been independently verified by human annotators and must **not** be used to claim official model capability.

### 5.1 Sensitivity Comparison

| Configuration | Historical Ground Truth Accuracy | AI-Proposed Labels Accuracy | Historical Macro-F1 | AI-Proposed Macro-F1 | Historical PFAR | AI-Proposed PFAR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 31.96% (62/194) | **51.03%** (99/194) | 0.2317 | 0.2974 | 80.00% | 80.00% |
| **Candidate B** | 33.51% (65/194) | **52.58%** (102/194) | 0.2491 | 0.3150 | 80.00% | 80.00% |

### 5.2 Critical Interpretation:
- The apparent 19% jump in accuracy (from ~33.5% to ~52.6%) is a mathematical artifact of collapsing 53 `SAFE_14_PLUS` samples into `SAFE_FOR_ALL`.
- Because the model previously predicted `SAFE_FOR_ALL` for almost all benign ads, relabeling the ground truth from 14+ to SFA turns previous model errors into "correct" predictions.
- **PFAR remains 80.00% identical**: None of the proposed label changes resolved the undetected scams or hazard footage.
