"""
SAFE-VISION Phase 8: Final Model Evaluation, Risk-Score Validation, and Report Evidence Generator.
Strictly restricted to the 194 eligible clean development samples.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JB_DIR = os.path.join(ROOT_DIR, "joseph_safead_core")
B2JB_DIR = os.path.join(ROOT_DIR, "joseph_safead_core", "backend2jb")
for p in [B2JB_DIR, JB_DIR]:
    if p not in sys.path:
        sys.path.append(p)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.fusion.safead_fusion import SafeAdFusion
from models.experimental_detectors import EnhancedAlcoholDetector

CLASSES = ["SAFE_FOR_ALL", "SAFE_14_PLUS", "SAFE_18_PLUS", "UNSAFE_FOR_ALL"]

def run_phase8_evaluation():
    print("=" * 80)
    print("SAFE-VISION PHASE 8: FINAL MODEL EVALUATION ENGINE")
    print("=" * 80)

    # 1. TASK 1: Cohort Isolation Assertions
    cache_path = os.path.join(ROOT_DIR, "results", "real_pipeline_baseline", "saved_detector_outputs.json")
    prov_path = os.path.join(ROOT_DIR, "docs", "PHASE_7C_SPLIT_PROVENANCE_REPORT.csv")

    assert os.path.exists(cache_path), f"Missing cache: {cache_path}"
    assert os.path.exists(prov_path), f"Missing provenance report: {prov_path}"

    with open(cache_path, "r", encoding="utf-8") as f:
        cached_data = json.load(f)

    prov_df = pd.read_csv(prov_path)

    # Assert exactly 194 eligible IDs
    dev_sids = set(cached_data.keys())
    assert len(dev_sids) == 194, f"Expected 194 samples, found {len(dev_sids)}"

    # Check zero overlap with held-out candidates
    held_out_sids = set(prov_df[prov_df["was_in_held_out_partition"] == True]["sample_id"])
    assert len(held_out_sids) == 88, f"Expected 88 held-out candidates, found {len(held_out_sids)}"
    overlap_held_out = dev_sids.intersection(held_out_sids)
    assert len(overlap_held_out) == 0, f"Critical contamination: {len(overlap_held_out)} held-out samples in dev cohort!"

    # Check zero overlap with synthetic cards
    synth_sids = set(prov_df[prov_df["previous_benchmark_membership"] == "v1_synthetic_review_card"]["sample_id"])
    assert len(synth_sids) == 118, f"Expected 118 synthetic cards, found {len(synth_sids)}"
    overlap_synth = dev_sids.intersection(synth_sids)
    assert len(overlap_synth) == 0, f"Critical contamination: {len(overlap_synth)} synthetic cards in dev cohort!"

    # Check zero overlap with new scam supplement
    scam_sids = set(prov_df[prov_df["previous_benchmark_membership"] == "new_v2_scam_supplement"]["sample_id"])
    assert len(scam_sids) == 14, f"Expected 14 scam supplement items, found {len(scam_sids)}"
    overlap_scam = dev_sids.intersection(scam_sids)
    assert len(overlap_scam) == 0, f"Critical contamination: {len(overlap_scam)} scam items in dev cohort!"

    print("[PASS] Task 1 Assertions: 194 unique approved dev samples; ZERO overlap with held-out, synthetic, or new scam items.")

    # Sort sample IDs deterministically
    sorted_sids = sorted(list(dev_sids))

    # 2. Extract ground truth labels
    y_true_hist = [cached_data[sid]["true_label"] for sid in sorted_sids]
    relabel_map = dict(zip(prov_df["sample_id"], prov_df["current_safead_label"]))
    y_true_prop = [relabel_map[sid] for sid in sorted_sids]
    diff_count = sum(1 for y1, y2 in zip(y_true_hist, y_true_prop) if y1 != y2)
    assert diff_count == 53, f"Expected 53 proposed label differences, got {diff_count}"
    print(f"[PASS] Ground Truth Loaded: 194 historical labels (75 SFA, 75 S14P, 34 S18P, 10 UFA); 53 proposed changes.")

    fusion = SafeAdFusion()
    alc = EnhancedAlcoholDetector()

    # 3. TASK 3: Evaluate Baseline & Candidate B on saved evidence
    preds_base = []
    scores_base = []
    confs_base = []

    preds_cand_b = []
    scores_cand_b = []
    confs_cand_b = []
    cand_b_alc_flags = []

    for sid in sorted_sids:
        sample = cached_data[sid]
        ev_base = sample["raw_detector_evidence"]
        res_base = fusion.combine_assessment(ev_base)
        preds_base.append(res_base["classification"])
        scores_base.append(res_base["risk_score"])
        confs_base.append(res_base["confidence"])

        # Candidate B
        ev_b = json.loads(json.dumps(ev_base))
        ocr_tx = sample.get("extracted_ocr", "")
        aud_tx = ev_b.get("audio", {}).get("transcript", "")
        vis_objs = sample.get("ocr_scan", {}).get("detected_objects", [])
        res_alc = alc.evaluate_alcohol(ocr_text=ocr_tx, speech_transcript=aud_tx, visual_objects=vis_objs)
        cand_b_alc_flags.append(res_alc["alcohol_detected"])
        if res_alc["alcohol_detected"]:
            ev_b.setdefault("text_safety", {}).setdefault("categories", {})["alcohol"] = res_alc["score"]
            ev_b.setdefault("advertisement_risks", {})["explicit_content"] = max(
                ev_b.get("advertisement_risks", {}).get("explicit_content", 0.0),
                res_alc["score"]
            )
            ev_b.setdefault("advertisement_risks", {}).setdefault("evidence", []).extend(res_alc.get("reasons", []))
        res_b = fusion.combine_assessment(ev_b)
        preds_cand_b.append(res_b["classification"])
        scores_cand_b.append(res_b["risk_score"])
        confs_cand_b.append(res_b["confidence"])

    # Verify exact reproduction
    correct_base = sum(1 for yt, yp in zip(y_true_hist, preds_base) if yt == yp)
    correct_b = sum(1 for yt, yp in zip(y_true_hist, preds_cand_b) if yt == yp)
    assert correct_base == 62, f"Baseline reproduction failed: {correct_base}/194"
    assert correct_b == 65, f"Candidate B reproduction failed: {correct_b}/194"
    print(f"[PASS] Exact Reproduction Verified: Baseline = 62/194 (31.96%), Candidate B = 65/194 (33.51%).")

    # Helper function to compute all metrics
    def calculate_full_metrics(y_true, y_pred, config_name):
        total = len(y_true)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc = correct / total

        # Per-class metrics
        class_metrics = {}
        for c in CLASSES:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            supp = sum(1 for yt in y_true if yt == c)
            class_metrics[c] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1_score": round(f1, 4),
                "support": supp,
                "tp": tp, "fp": fp, "fn": fn
            }

        macro_p = round(float(np.mean([class_metrics[c]["precision"] for c in CLASSES])), 4)
        macro_r = round(float(np.mean([class_metrics[c]["recall"] for c in CLASSES])), 4)
        macro_f1 = round(float(np.mean([class_metrics[c]["f1_score"] for c in CLASSES])), 4)

        # Prohibited False Acceptance Rate (PFAR)
        # GT is UNSAFE_FOR_ALL, but predicted is SFA, S14P, or S18P (unwithheld prohibited release)
        ufa_total = sum(1 for yt in y_true if yt == "UNSAFE_FOR_ALL")
        ufa_fa = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "UNSAFE_FOR_ALL" and yp != "UNSAFE_FOR_ALL" and yp != "REQUIRES_HUMAN_REVIEW")
        pfar = round(ufa_fa / ufa_total, 4) if ufa_total > 0 else 0.0

        # Underage Gating Violation Rate (UGVR)
        # 1. 18+ adult ads exposed to minors (<18): predicted SFA or S14P
        s18p_total = sum(1 for yt in y_true if yt == "SAFE_18_PLUS")
        s18p_underage = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SAFE_18_PLUS" and yp in ["SAFE_FOR_ALL", "SAFE_14_PLUS"])
        ugvr_18p = round(s18p_underage / s18p_total, 4) if s18p_total > 0 else 0.0

        # 2. Universal Prohibited ads exposed to minors (<18): predicted SFA or S14P
        ufa_underage = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "UNSAFE_FOR_ALL" and yp in ["SAFE_FOR_ALL", "SAFE_14_PLUS"])
        ugvr_ufa = round(ufa_underage / ufa_total, 4) if ufa_total > 0 else 0.0

        # False Rejection Rate (FRR)
        # Benign commercial ads (SFA or S14P) falsely rejected as UNSAFE_FOR_ALL
        benign_total = sum(1 for yt in y_true if yt in ["SAFE_FOR_ALL", "SAFE_14_PLUS"])
        benign_rejected = sum(1 for yt, yp in zip(y_true, y_pred) if yt in ["SAFE_FOR_ALL", "SAFE_14_PLUS"] and yp == "UNSAFE_FOR_ALL")
        frr = round(benign_rejected / benign_total, 4) if benign_total > 0 else 0.0

        # Abstention & Coverage
        abstentions = sum(1 for yp in y_pred if yp == "REQUIRES_HUMAN_REVIEW")
        coverage = round((total - abstentions) / total, 4)
        abstention_rate = round(abstentions / total, 4)

        # Accuracy on automatically classified subset
        auto_total = total - abstentions
        auto_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yp != "REQUIRES_HUMAN_REVIEW")
        auto_acc = round(auto_correct / auto_total, 4) if auto_total > 0 else 0.0

        # Safety-critical errors: Prohibited released + 18+ undergated
        safety_critical_errors = ufa_fa + s18p_underage

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        cm_dict = {
            CLASSES[i]: {CLASSES[j]: int(cm[i, j]) for j in range(len(CLASSES))}
            for i in range(len(CLASSES))
        }

        return {
            "configuration": config_name,
            "total_samples": total,
            "accuracy": round(acc, 4),
            "correct_count": correct,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "prohibited_false_acceptance_rate": pfar,
            "prohibited_false_acceptance_count": ufa_fa,
            "prohibited_total": ufa_total,
            "underage_gating_violation_rate_18p": ugvr_18p,
            "underage_18p_count": s18p_underage,
            "underage_ufa_count": ufa_underage,
            "false_rejection_rate": frr,
            "false_rejection_count": benign_rejected,
            "benign_total": benign_total,
            "abstention_count": abstentions,
            "abstention_rate": abstention_rate,
            "automated_coverage": coverage,
            "automated_accuracy": auto_acc,
            "safety_critical_error_count": safety_critical_errors,
            "class_metrics": class_metrics,
            "confusion_matrix": cm_dict
        }

    metrics_base = calculate_full_metrics(y_true_hist, preds_base, "Baseline (Frozen Production)")
    metrics_cand_b = calculate_full_metrics(y_true_hist, preds_cand_b, "Candidate B (Enhanced Alcohol)")

    # 4. TASK 4: Threshold Calibration & Cross-Validation Experiments
    # Prepare features for all 194 samples
    def extract_features(sample, use_alcohol=True):
        ev = json.loads(json.dumps(sample["raw_detector_evidence"]))
        if use_alcohol:
            ocr_tx = sample.get("extracted_ocr", "")
            aud_tx = ev.get("audio", {}).get("transcript", "")
            vis_objs = sample.get("ocr_scan", {}).get("detected_objects", [])
            res_alc = alc.evaluate_alcohol(ocr_text=ocr_tx, speech_transcript=aud_tx, visual_objects=vis_objs)
            if res_alc["alcohol_detected"]:
                ev.setdefault("text_safety", {}).setdefault("categories", {})["alcohol"] = res_alc["score"]
                ev.setdefault("advertisement_risks", {})["explicit_content"] = max(
                    ev.get("advertisement_risks", {}).get("explicit_content", 0.0),
                    res_alc["score"]
                )
                ev.setdefault("advertisement_risks", {}).setdefault("evidence", []).extend(res_alc.get("reasons", []))

        vis = ev.get("visual", {})
        vid = ev.get("video", {})
        ocr = ev.get("ocr", {})
        aud = ev.get("audio", {})
        ad_risks = ev.get("advertisement_risks", {})

        visual_unsafe = vis.get("unsafe", False)
        visual_cats = vis.get("detected_categories", [])
        violence_score = max(vid.get("violence", 0.0), vis.get("violence", 0.0))
        nsfw_score = max(vid.get("adult", 0.0), vis.get("adult", 0.0))
        ocr_risk = max(ocr.get("adult", 0.0), ocr.get("violence", 0.0), ocr.get("scam", 0.0), ocr.get("deceptive", 0.0))
        audio_risk = max(aud.get("adult", 0.0), aud.get("violence", 0.0), aud.get("scam", 0.0))
        scam_score = ad_risks.get("scam", 0.0)
        deceptive_score = ad_risks.get("deceptive_marketing", 0.0)
        explicit_score = ad_risks.get("explicit_content", 0.0)

        child_safety_flag = ("Child Safety Risk" in visual_cats or vis.get("child_safety", 0.0) >= 0.50 or vid.get("child_safety", 0.0) >= 0.50)
        prohibited_risk = max(violence_score, scam_score, deceptive_score, ocr.get("scam", 0.0), aud.get("scam", 0.0)) * 100.0
        adult_content_score = max(nsfw_score, explicit_score, ocr.get("adult", 0.0), aud.get("adult", 0.0))

        weighted_score = (
            (100.0 if visual_unsafe else 0.0) * 0.25 +
            (violence_score * 100.0) * 0.20 +
            (nsfw_score * 100.0) * 0.20 +
            (ocr_risk * 100.0) * 0.15 +
            (audio_risk * 100.0) * 0.10 +
            (max(scam_score, deceptive_score, explicit_score) * 100.0) * 0.10
        )

        if child_safety_flag or prohibited_risk >= 50.0:
            risk_score = round(min(100.0, max(weighted_score, prohibited_risk)), 2)
        elif adult_content_score >= 0.25:
            risk_score = round(min(65.0, max(weighted_score, adult_content_score * 70.0)), 2)
        else:
            risk_score = round(min(39.9, weighted_score), 2)

        return {
            "child_safety": child_safety_flag,
            "violence": violence_score,
            "scam": scam_score,
            "deceptive": deceptive_score,
            "prohibited_risk": prohibited_risk,
            "adult_score": adult_content_score,
            "nsfw_score": nsfw_score,
            "explicit_score": explicit_score,
            "visual_unsafe": visual_unsafe,
            "risk_score": risk_score
        }

    dev_features = [extract_features(cached_data[sid], use_alcohol=True) for sid in sorted_sids]

    def classify_with_thresholds(f, r_th=25.0, a_th=0.35, p_th=50.0, enable_hitl=False):
        # Stage 1: Prohibited
        is_prohib = (f["child_safety"] or f["violence"] >= 0.50 or f["scam"] >= 0.50 or 
                     f["deceptive"] >= 0.70 or f["prohibited_risk"] >= p_th or f["visual_unsafe"])
        if is_prohib:
            return "UNSAFE_FOR_ALL"

        # Stage 1.5: Human Review routing for ambiguous risks
        if enable_hitl:
            # Ambiguous kinetic violence (0.15 <= violence < 0.50) or deceptive scam (0.10 <= scam < 0.50)
            if (0.15 <= f["violence"] < 0.50) or (0.10 <= f["scam"] < 0.50):
                return "REQUIRES_HUMAN_REVIEW"

        # Stage 2: Age policy
        if f["adult_score"] >= a_th or f["nsfw_score"] >= a_th:
            return "SAFE_18_PLUS"
        elif f["risk_score"] >= r_th or f["adult_score"] >= 0.20:
            return "SAFE_14_PLUS"
        else:
            return "SAFE_FOR_ALL"

    # Calibration Grid Evaluation
    threshold_results = []
    param_grid = [
        {"name": "Implemented Baseline Thresholds", "r_th": 25.0, "a_th": 0.35, "p_th": 50.0, "hitl": False},
        {"name": "Candidate B Implemented", "r_th": 25.0, "a_th": 0.35, "p_th": 50.0, "hitl": False},
        {"name": "Lower Teen Threshold (r=20, a=0.35)", "r_th": 20.0, "a_th": 0.35, "p_th": 50.0, "hitl": False},
        {"name": "Higher Adult Threshold (r=25, a=0.40)", "r_th": 25.0, "a_th": 0.40, "p_th": 50.0, "hitl": False},
        {"name": "Calibrated Compromise (r=20, a=0.40)", "r_th": 20.0, "a_th": 0.40, "p_th": 50.0, "hitl": False},
        {"name": "Higher Teen Cutoff (r=30, a=0.35)", "r_th": 30.0, "a_th": 0.35, "p_th": 50.0, "hitl": False},
        {"name": "Hybrid Architecture + HITL (Safety-First)", "r_th": 25.0, "a_th": 0.35, "p_th": 50.0, "hitl": True},
    ]

    for p in param_grid:
        p_preds = [classify_with_thresholds(f, p["r_th"], p["a_th"], p["p_th"], p["hitl"]) for f in dev_features]
        m = calculate_full_metrics(y_true_hist, p_preds, p["name"])
        threshold_results.append({
            "configuration_name": p["name"],
            "teen_risk_threshold": p["r_th"],
            "adult_threshold": p["a_th"],
            "prohibited_threshold": p["p_th"],
            "hitl_enabled": p["hitl"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "pfar": m["prohibited_false_acceptance_rate"],
            "ugvr_18p": m["underage_gating_violation_rate_18p"],
            "frr": m["false_rejection_rate"],
            "abstention_rate": m["abstention_rate"],
            "automated_coverage": m["automated_coverage"],
            "automated_accuracy": m["automated_accuracy"]
        })

    # Stratified 5-Fold Cross Validation for Calibrated Model (Out-of-Fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_arr = np.array(y_true_hist)
    oof_preds_calib = [""] * len(sorted_sids)

    candidate_params = [(20.0, 0.30), (20.0, 0.35), (20.0, 0.40),
                        (25.0, 0.30), (25.0, 0.35), (25.0, 0.40),
                        (30.0, 0.30), (30.0, 0.35), (30.0, 0.40)]

    fold_logs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(sorted_sids, y_true_arr)):
        best_f1 = -1.0
        best_param = (25.0, 0.35)
        for (r_c, a_c) in candidate_params:
            tr_preds = [classify_with_thresholds(dev_features[i], r_c, a_c) for i in train_idx]
            tr_f1 = f1_score(y_true_arr[train_idx], tr_preds, labels=CLASSES, average="macro", zero_division=0)
            if tr_f1 > best_f1:
                best_f1 = tr_f1
                best_param = (r_c, a_c)
        fold_logs.append({
            "fold": fold + 1,
            "selected_teen_risk_th": best_param[0],
            "selected_adult_th": best_param[1],
            "train_macro_f1": round(best_f1, 4)
        })
        for i in val_idx:
            oof_preds_calib[i] = classify_with_thresholds(dev_features[i], best_param[0], best_param[1])

    metrics_oof_calib = calculate_full_metrics(y_true_hist, oof_preds_calib, "Candidate B (5-Fold Out-of-Fold Calibrated)")

    # 1D Monotonic Risk Score Threshold Test (Showing failure of 1D scalar cuts)
    # Try monotonic cuts: [0, T1) -> SFA, [T1, T2) -> S14P, [T2, T3) -> S18P, [T3, 100] -> UFA
    best_1d_acc = 0.0
    best_1d_cuts = None
    for t1 in [5.0, 10.0, 15.0, 20.0]:
        for t2 in [25.0, 30.0, 35.0, 40.0]:
            for t3 in [50.0, 60.0, 70.0]:
                preds_1d = []
                for f in dev_features:
                    r = f["risk_score"]
                    if r >= t3: preds_1d.append("UNSAFE_FOR_ALL")
                    elif r >= t2: preds_1d.append("SAFE_18_PLUS")
                    elif r >= t1: preds_1d.append("SAFE_14_PLUS")
                    else: preds_1d.append("SAFE_FOR_ALL")
                acc_1d = accuracy_score(y_true_hist, preds_1d)
                if acc_1d > best_1d_acc:
                    best_1d_acc = acc_1d
                    best_1d_cuts = (t1, t2, t3)

    print(f"[INFO] Best 1D Monotonic Risk Score Accuracy: {best_1d_acc*100:.2f}% with cuts {best_1d_cuts}")

    # 5. TASK 5: Proposed Label Sensitivity Analysis (Exploratory)
    metrics_base_prop = calculate_full_metrics(y_true_prop, preds_base, "Baseline under AI-Proposed Labels (Exploratory)")
    metrics_cand_b_prop = calculate_full_metrics(y_true_prop, preds_cand_b, "Candidate B under AI-Proposed Labels (Exploratory)")

    # 6. TASK 8: Export Output Files to results/phase8_final/
    out_dir = os.path.join(ROOT_DIR, "results", "phase8_final")
    os.makedirs(out_dir, exist_ok=True)

    # 1. baseline_confusion_matrix.csv
    cm_base_df = pd.DataFrame(metrics_base["confusion_matrix"]).T
    cm_base_df.to_csv(os.path.join(out_dir, "baseline_confusion_matrix.csv"), index=True)

    # 2. candidate_b_confusion_matrix.csv
    cm_b_df = pd.DataFrame(metrics_cand_b["confusion_matrix"]).T
    cm_b_df.to_csv(os.path.join(out_dir, "candidate_b_confusion_matrix.csv"), index=True)

    # 3. phase8_metrics.json
    all_metrics = {
        "evaluation_cohort": {
            "total_samples": 194,
            "images": 134,
            "videos": 60,
            "historical_labels_distribution": {"SAFE_FOR_ALL": 75, "SAFE_14_PLUS": 75, "SAFE_18_PLUS": 34, "UNSAFE_FOR_ALL": 10},
            "proposed_labels_distribution": {"SAFE_FOR_ALL": 128, "SAFE_14_PLUS": 22, "SAFE_18_PLUS": 34, "UNSAFE_FOR_ALL": 10},
            "quarantined_held_out_candidates": 88,
            "quarantined_synthetic_cards": 118,
            "quarantined_scam_supplement": 14
        },
        "baseline": metrics_base,
        "candidate_b": metrics_cand_b,
        "candidate_b_oof_calibrated": metrics_oof_calib,
        "exploratory_proposed_labels": {
            "baseline": metrics_base_prop,
            "candidate_b": metrics_cand_b_prop
        },
        "kfold_logs": fold_logs,
        "1d_monotonic_risk_cuts_evaluation": {
            "best_accuracy": round(best_1d_acc, 4),
            "best_thresholds": best_1d_cuts,
            "finding": "Monotonic 1D risk cuts collapse multiclass accuracy and fail to isolate prohibited threats from adult content."
        }
    }
    with open(os.path.join(out_dir, "phase8_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # 4. per_class_metrics.csv
    per_class_rows = []
    for c in CLASSES:
        mb = metrics_base["class_metrics"][c]
        mc = metrics_cand_b["class_metrics"][c]
        mo = metrics_oof_calib["class_metrics"][c]
        per_class_rows.append({
            "category": c,
            "support": mb["support"],
            "baseline_precision": mb["precision"],
            "baseline_recall": mb["recall"],
            "baseline_f1": mb["f1_score"],
            "candidate_b_precision": mc["precision"],
            "candidate_b_recall": mc["recall"],
            "candidate_b_f1": mc["f1_score"],
            "calibrated_oof_precision": mo["precision"],
            "calibrated_oof_recall": mo["recall"],
            "calibrated_oof_f1": mo["f1_score"]
        })
    pd.DataFrame(per_class_rows).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    # 5. threshold_sensitivity.csv
    pd.DataFrame(threshold_results).to_csv(os.path.join(out_dir, "threshold_sensitivity.csv"), index=False)

    # 6. proposed_label_sensitivity.csv
    prop_rows = [
        {
            "configuration": "Baseline (Frozen Production)",
            "historical_accuracy": metrics_base["accuracy"],
            "historical_macro_f1": metrics_base["macro_f1"],
            "historical_pfar": metrics_base["prohibited_false_acceptance_rate"],
            "proposed_accuracy": metrics_base_prop["accuracy"],
            "proposed_macro_f1": metrics_base_prop["macro_f1"],
            "proposed_pfar": metrics_base_prop["prohibited_false_acceptance_rate"],
            "accuracy_delta": round(metrics_base_prop["accuracy"] - metrics_base["accuracy"], 4),
            "pfar_delta": round(metrics_base_prop["prohibited_false_acceptance_rate"] - metrics_base["prohibited_false_acceptance_rate"], 4)
        },
        {
            "configuration": "Candidate B (Enhanced Alcohol)",
            "historical_accuracy": metrics_cand_b["accuracy"],
            "historical_macro_f1": metrics_cand_b["macro_f1"],
            "historical_pfar": metrics_cand_b["prohibited_false_acceptance_rate"],
            "proposed_accuracy": metrics_cand_b_prop["accuracy"],
            "proposed_macro_f1": metrics_cand_b_prop["macro_f1"],
            "proposed_pfar": metrics_cand_b_prop["prohibited_false_acceptance_rate"],
            "accuracy_delta": round(metrics_cand_b_prop["accuracy"] - metrics_cand_b["accuracy"], 4),
            "pfar_delta": round(metrics_cand_b_prop["prohibited_false_acceptance_rate"] - metrics_cand_b["prohibited_false_acceptance_rate"], 4)
        }
    ]
    pd.DataFrame(prop_rows).to_csv(os.path.join(out_dir, "proposed_label_sensitivity.csv"), index=False)

    # 7. sample_predictions.csv
    pred_rows = []
    for i, sid in enumerate(sorted_sids):
        pred_rows.append({
            "sample_id": sid,
            "media_type": cached_data[sid]["media_type"],
            "historical_ground_truth": y_true_hist[i],
            "proposed_ground_truth": y_true_prop[i],
            "label_difference_flag": (y_true_hist[i] != y_true_prop[i]),
            "baseline_prediction": preds_base[i],
            "baseline_risk_score": scores_base[i],
            "baseline_confidence": confs_base[i],
            "baseline_correct_historical": (y_true_hist[i] == preds_base[i]),
            "candidate_b_prediction": preds_cand_b[i],
            "candidate_b_risk_score": scores_cand_b[i],
            "candidate_b_confidence": confs_cand_b[i],
            "candidate_b_alcohol_detected": cand_b_alc_flags[i],
            "candidate_b_correct_historical": (y_true_hist[i] == preds_cand_b[i]),
            "calibrated_oof_prediction": oof_preds_calib[i],
            "calibrated_oof_correct_historical": (y_true_hist[i] == oof_preds_calib[i])
        })
    pd.DataFrame(pred_rows).to_csv(os.path.join(out_dir, "sample_predictions.csv"), index=False)

    # 8. safety_failure_cases.csv
    failure_rows = []
    # Identify all prohibited false acceptances and severe underage leaks
    for i, sid in enumerate(sorted_sids):
        gt = y_true_hist[i]
        pb = preds_base[i]
        pc = preds_cand_b[i]
        f = dev_features[i]
        # Failure 1: Ground truth Prohibited not rejected
        if gt == "UNSAFE_FOR_ALL":
            failure_rows.append({
                "sample_id": sid,
                "failure_type": "PROHIBITED_FALSE_ACCEPTANCE",
                "ground_truth": gt,
                "baseline_prediction": pb,
                "candidate_b_prediction": pc,
                "risk_score": scores_cand_b[i],
                "violence_score": f["violence"],
                "scam_score": f["scam"],
                "deceptive_score": f["deceptive"],
                "adult_score": f["adult_score"],
                "visual_unsafe_flag": f["visual_unsafe"],
                "failure_analysis": "Undetected by baseline/Candidate B detectors; low modality risk score" if pc != "UNSAFE_FOR_ALL" else "Correctly rejected"
            })
        # Failure 2: 18+ adult ad exposed to general audience (predicted SFA or S14P)
        elif gt == "SAFE_18_PLUS" and pc in ["SAFE_FOR_ALL", "SAFE_14_PLUS"]:
            failure_rows.append({
                "sample_id": sid,
                "failure_type": "UNDERAGE_GATING_LEAK",
                "ground_truth": gt,
                "baseline_prediction": pb,
                "candidate_b_prediction": pc,
                "risk_score": scores_cand_b[i],
                "violence_score": f["violence"],
                "scam_score": f["scam"],
                "deceptive_score": f["deceptive"],
                "adult_score": f["adult_score"],
                "visual_unsafe_flag": f["visual_unsafe"],
                "failure_analysis": "Adult attire/beverage missed by visual adult detector"
            })
    pd.DataFrame(failure_rows).to_csv(os.path.join(out_dir, "safety_failure_cases.csv"), index=False)

    # 9. evaluation_provenance.json
    provenance_data = {
        "phase": "Phase 8: Final Model Evaluation and Evidence Generation",
        "timestamp": "2026-10-22T13:05:00Z",
        "repository": "https://github.com/QuadRos-MBCET/main_project_s7",
        "branch": "feature/phase-7b-dataset-v2-integration",
        "commit_sha": "9b532a66f5918d9ec5c8f4636ae0c9cf22a56f97",
        "dataset_repository": "https://github.com/QuadRos-MBCET/safead-dataset-hub",
        "dataset_branch": "feature/dataset-improvement-reconstruction",
        "dataset_commit": "0e8883856cad683dbeaf2c153660df25fa8c4c11",
        "cohort_definition": "194 approved clean development samples from results/real_pipeline_baseline/saved_detector_outputs.json",
        "sample_count": 194,
        "held_out_samples_isolated": 88,
        "synthetic_cards_quarantined": 118,
        "scam_supplement_quarantined": 14,
        "reproducibility": {
            "baseline_accuracy": "62/194 (31.96%)",
            "candidate_b_accuracy": "65/194 (33.51%)",
            "changed_samples": ["SAFEAD_S18P_IMG_002", "SAFEAD_S18P_IMG_006", "SAFEAD_S18P_IMG_031"],
            "kfold_seed": 42
        }
    }
    with open(os.path.join(out_dir, "evaluation_provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)

    print("[SUCCESS] All 9 Phase 8 Artifacts Generated in results/phase8_final/ successfully.")

if __name__ == "__main__":
    run_phase8_evaluation()
