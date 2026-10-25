# SAFE-VISION — Phase 8: Report-Ready Results & Academic Evidence Compendium

**Project**: SAFE-VISION (SafeAd AI Multimodal Advertisement Moderation System)  
**Document**: Report-Ready Evidence, Tables, Equations, and Acceptance Documentation  
**Date**: October 22, 2026  
**Status**: COMPLETE & VERIFIED  

---

## 1. Project Context & Academic Defense

This document compiles the empirical results, mathematical specifications, regulatory justifications, and safety disclosures required for the final SAFE-VISION academic report.

### Core Architecture Summary:
SAFE-VISION is an automated multimodal advertising safety and age-gating system. It processes image, video, audio transcripts, and OCR overlays through a **Two-Stage Hybrid Architecture**:
1. **Stage 1 (Prohibited Content Policy Engine)**: Enforces hard policy checks for unlawful scams, child exploitation, extreme graphic violence, and hate speech.
2. **Stage 2 (Age Policy Engine)**: Evaluates non-prohibited commercial inventory for age-appropriate delivery across three demographic tiers (`SAFE_FOR_ALL`, `SAFE_14_PLUS`, `SAFE_18_PLUS`).

---

## 2. Definitive Numerical Results Summary

All results are computed on the **194 approved clean development samples** from `results/real_pipeline_baseline/saved_detector_outputs.json` with **zero test set contamination**.

### 2.1 Benchmark Performance Across Configurations

| Metric | Production Baseline | Candidate B (Enhanced Alcohol) | 5-Fold OOF Calibrated | Safety Hybrid + HITL | Candidate B under AI-Proposed Labels* |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | 31.96% (62/194) | **33.51% (65/194)** | 34.02% (66/194) | 24.23% (47/194) | 52.58% (102/194)* |
| **Macro Precision** | 26.82% | **28.25%** | 27.06% | 27.53% | 33.38%* |
| **Macro Recall** | 28.62% | **30.82%** | 30.35% | 23.36% | 32.89%* |
| **Macro F1-Score** | 0.2317 | **0.2491** | 0.2575 | 0.2020 | 0.3150* |
| **SAFE_FOR_ALL F1** | 0.4974 | **0.5053** | 0.5053 | 0.4419 | 0.7078* |
| **SAFE_14_PLUS F1** | 0.0723 | **0.0723** | 0.1277 | 0.0741 | 0.1333* |
| **SAFE_18_PLUS F1** | 0.2143 | **0.2759** | 0.2632 | 0.2105 | 0.2759* |
| **UNSAFE_FOR_ALL F1** | 0.1429 | **0.1429** | 0.1429 | 0.0816 | 0.1429* |
| **Prohibited False Acceptance (PFAR)** | 80.00% (8/10) | **80.00% (8/10)** | 80.00% (8/10) | **70.00% (7/10)** | 80.00% (8/10)* |
| **Underage Gating Violations (18+ -> <18)**| 58.82% (20/34) | **50.00% (17/34)** | 52.94% (18/34) | **26.47% (9/34)** | 50.00% (17/34)* |
| **False Rejection Rate (Benign -> UFA)** | 7.33% (11/150) | **7.33% (11/150)** | 7.33% (11/150) | 7.33% (11/150) | 7.33% (11/150)* |
| **Automated Coverage / Throughput** | 100.0% (194/194) | **100.0% (194/194)** | 100.0% (194/194) | **77.84% (151/194)**| 100.0% (194/194)* |
| **Automated Accuracy** | 31.96% | **33.51%** | 34.02% | **31.13%** | 52.58%* |

*\* Exploratory sensitivity analysis on unverified AI-proposed label changes — NOT validated ground-truth accuracy.*

---

## 3. Executive Answers to Core Project Questions

### 1. Actual Risk-Score Equation & Coefficients
$$\begin{aligned}
S_{\text{weighted}} = &\; 0.25 \times (100.0 \cdot \mathbf{v}_{\text{unsafe}}) + 0.20 \times (100.0 \cdot s_{\text{viol}}) + 0.20 \times (100.0 \cdot s_{\text{nsfw}}) \\
&+ 0.15 \times (100.0 \cdot s_{\text{ocr}}) + 0.10 \times (100.0 \cdot s_{\text{aud}}) + 0.10 \times (100.0 \cdot s_{\text{adrisk}})
\end{aligned}$$
Non-linear piecewise scaling:
$$S_{\text{risk}} = \begin{cases} 
\min(100.0, \max(S_{\text{weighted}}, s_{\text{prohibited}})), & \text{if } f_{\text{child}} \text{ or } s_{\text{prohibited}} \ge 50.0 \\
\min(65.0, \max(S_{\text{weighted}}, s_{\text{adult}} \times 70.0)), & \text{else if } s_{\text{adult}} \ge 0.25 \\
\min(39.9, S_{\text{weighted}}), & \text{otherwise}
\end{cases}$$

### 2. Existing Implemented Thresholds
- **Stage 1 (Prohibited)**: $s_{\text{viol}} \ge 0.50$, $s_{\text{scam}} \ge 0.50$, $s_{\text{decep}} \ge 0.70$, $s_{\text{prohibited}} \ge 50.0$.
- **Stage 2 (Age Policy)**: $s_{\text{adult}} \ge 0.35 \implies \text{SAFE\_18\_PLUS}$; $S_{\text{risk}} \ge 25.0 \implies \text{SAFE\_14\_PLUS}$; Else $\implies \text{SAFE\_FOR\_ALL}$.
- **Human-in-the-Loop**: $\text{confidence} < 0.50$ (In practice, never triggers because confidence formula evaluates to $\ge 0.94$).

### 3. Experimentally Evaluated Candidate Thresholds
- Evaluated grid: $T_{\text{teen}} \in \{20.0, 25.0, 30.0\}$, $T_{\text{adult}} \in \{0.30, 0.35, 0.40\}$.
- **OOF Calibrated Configuration**: $T_{\text{teen}} = 20.0, T_{\text{adult}} = 0.40$. Yields 66/194 accuracy (34.02%), Macro-F1 0.2575.
- **Safety Hybrid + HITL Configuration**: Routes ambiguous signals ($0.15 \le s_{\text{viol}} < 0.50$ or $0.10 \le s_{\text{scam}} < 0.50$) to Moderator Review. Cuts underage violations from 50% to 26.47%.

### 4. Best Defensible Model Configuration
- **For Academic Performance Reporting**: **Candidate B** (Baseline + `EnhancedAlcoholDetector`) with implemented thresholds ($T_{\text{teen}}=25.0, T_{\text{adult}}=0.35$). Demonstrates clear, traceable gain (+3 samples, +8.82% 18+ recall) on real commercial beverages with zero non-alcoholic false positives.
- **For Production System Safety**: **Safety-First Hybrid with HITL Routing**. Essential to mitigate severe liability from undetected prohibited scams.

### 5. Baseline Accuracy & Macro-F1
- **Accuracy**: **31.96%** (62/194).
- **Macro-F1**: **0.2317** (Macro-P: 26.82%, Macro-R: 28.62%).

### 6. Candidate B Accuracy & Macro-F1
- **Accuracy**: **33.51%** (65/194).
- **Macro-F1**: **0.2491** (Macro-P: 28.25%, Macro-R: 30.82%).

### 7. Threshold-Calibrated Results
- **OOF Accuracy**: **34.02%** (66/194).
- **OOF Macro-F1**: **0.2575**.
- **Status**: Reported as **provisional development calibration** due to small prohibited sample size ($N=10$).

### 8. Prohibited False Acceptance & Underage Violation Rates
- **PFAR**: **80.00%** (8 out of 10 prohibited ads released to users).
- **Underage Gating Violations (18+ -> <18)**: **50.00%** under Candidate B (17/34) vs **58.82%** under Baseline (20/34).
- **Safety Hybrid with HITL**: Reduces PFAR to **70.00%** and underage violations to **26.47%**.

### 9. Exploratory Label Sensitivity Results
- **Baseline under Proposed Labels**: Accuracy: **51.03%** (99/194), Macro-F1: **0.2974**.
- **Candidate B under Proposed Labels**: Accuracy: **52.58%** (102/194), Macro-F1: **0.3150**.
- **PFAR**: **80.00%** (Completely unchanged).
- **Nature**: Exploratory analysis showing how unverified label changes artificially inflate accuracy by reclassifying misclassified 14+ ads as SFA.

### 10. Authoritative Policy References with URLs
1. **International Age Rating Coalition (IARC)**: Rating Definitions & Content Descriptors  
   [https://www.globalratings.com/ratings-guide.aspx](https://www.globalratings.com/ratings-guide.aspx)
2. **Google Ads Policies**: Prohibited & Restricted Advertising Standards  
   [https://support.google.com/adspolicy/answer/6008942](https://support.google.com/adspolicy/answer/6008942)  
   [https://support.google.com/adspolicy/answer/6015406](https://support.google.com/adspolicy/answer/6015406) (Prohibited)  
   [https://support.google.com/adspolicy/answer/6018017](https://support.google.com/adspolicy/answer/6018017) (Restricted/Alcohol)
3. **Meta Advertising Standards**: Unacceptable & Restricted Content  
   [https://transparency.meta.com/policies/ad-standards/](https://transparency.meta.com/policies/ad-standards/)  
   [https://transparency.meta.com/policies/ad-standards/unacceptable-content/](https://transparency.meta.com/policies/ad-standards/unacceptable-content/)  
   [https://transparency.meta.com/policies/ad-standards/restricted-goods-services/](https://transparency.meta.com/policies/ad-standards/restricted-goods-services/)
4. **FTC Deception Policy Statement**: Section 5 Three-Part Deception Test  
   [https://www.ftc.gov/legal-library/browse/ftc-policy-statement-on-deception](https://www.ftc.gov/legal-library/browse/ftc-policy-statement-on-deception)
5. **NIST AI Risk Management Framework (AI RMF 1.0)**: Trustworthy AI & Governance  
   [https://www.nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
6. **ISO 31000:2018**: Risk Management Guidelines & Criteria  
   [https://www.iso.org/standard/65694.html](https://www.iso.org/standard/65694.html)

### 11. Report-Ready Files Generated
- `results/phase8_final/baseline_confusion_matrix.csv`
- `results/phase8_final/candidate_b_confusion_matrix.csv`
- `results/phase8_final/phase8_metrics.json`
- `results/phase8_final/per_class_metrics.csv`
- `results/phase8_final/threshold_sensitivity.csv`
- `results/phase8_final/proposed_label_sensitivity.csv`
- `results/phase8_final/sample_predictions.csv`
- `results/phase8_final/evaluation_provenance.json`
- `results/phase8_final/safety_failure_cases.csv`
- `docs/PHASE_8_FINAL_MODEL_EVALUATION.md`
- `docs/RISK_SCORE_EQUATION_AND_THRESHOLD_VALIDATION.md`
- `docs/FOUR_CATEGORY_POLICY_JUSTIFICATION.md`
- `docs/PHASE_8_REPORT_READY_RESULTS.md`

### 12. Unresolved Limitations
1. **Severe Prohibited False Acceptance (80%)**: Current neural backbones fail to detect investment scams, cryptocurrency fraud, and non-violent illegal offers without fine-tuning.
2. **Adolescent Class (`SAFE_14_PLUS`) Collapsed Recall (4.0%)**: Generic multimodal features cannot distinguish PG-13 action combat and high-stimulant energy drinks from general commercial items.
3. **Small Sample Size for Prohibited Content ($N=10$)**: Evaluated development set contains only 10 prohibited ads, limiting cross-validation statistical power.
4. **Heuristic Confidence**: The model confidence score reflects modality completeness rather than calibrated uncertainty.

### 13. Definitive Recommendation for the Project Report
- **GO FOR ACADEMIC REPORT PUBLICATION**: The evidence, mathematical equations, cross-validation protocols, and failure analyses are fully verified, reproducible, and defensible.
- **NO-GO FOR AUTOMATIC PRODUCTION PUBLICATION**: With an 80% prohibited false acceptance rate and 50% underage gating leakage, the automated system cannot operate safely without human moderator oversight.
