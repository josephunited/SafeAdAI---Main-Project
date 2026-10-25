import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
except Exception:
    pass

import time
from typing import Dict, Any, List, Optional
from PIL import Image


from processing.preprocessing import validate_advertisement_file
from processing.video_sampling import extract_adaptive_keyframes, get_video_metadata
from processing.audio_extraction import extract_audio_to_wav, cleanup_audio_file

# Backward-compatibility aliases for existing test suites & safety pipeline modules
validate_advertisement_input = validate_advertisement_file
sample_video_frames = extract_adaptive_keyframes
load_video_metadata = get_video_metadata

def print_video_processing_report(video_path: str, frames: list, metadata: dict = None) -> None:
    meta = metadata or get_video_metadata(video_path)
    print(f"[SafeAd AI Video Report] File: {os.path.basename(video_path)} | Frames Extracted: {len(frames)} | FPS: {meta.get('fps', 0)} | Duration: {meta.get('duration', 0)}s")


from models.model_manager import model_manager
from models.visual_safety import VisualSafetyDetector
from models.nsfw_detector import SafeAdNSFWDetector
from models.violence_detector import SafeAdViolenceDetector
from models.ocr_service import SafeAdOCRService
from models.audio_service import SafeAdAudioService
from models.text_safety import SafeAdTextSafetyDetector
from models.ad_risk_analyzer import AdRiskAnalyzer
from models.fusion.safead_assessment import SafeAdAssessment
from models.fusion.safead_fusion import SafeAdFusion

from models.experimental_detectors import EnhancedAlcoholDetector, KineticViolenceDecoupler, LearnedRFFusionClassifier

# Singleton instances for lazy invocation
visual_detector = VisualSafetyDetector()
nsfw_detector = SafeAdNSFWDetector()
violence_detector = SafeAdViolenceDetector()
ocr_service = SafeAdOCRService()
audio_service = SafeAdAudioService()
text_safety_detector = SafeAdTextSafetyDetector()
ad_risk_analyzer = AdRiskAnalyzer()
fusion_engine = SafeAdFusion()

# Experimental detector singletons (active ONLY when explicitly selected)
experimental_alcohol_detector = EnhancedAlcoholDetector()
experimental_violence_decoupler = KineticViolenceDecoupler()
experimental_learned_rf = LearnedRFFusionClassifier()

def run_safead_inference(
    file_path: str,
    title: str = "",
    caption: str = "",
    pipeline_mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end SafeAd AI multimodal advertisement moderation pipeline.
    Synthesizes evidence into Central SafeAdAssessment Matrix before SafeAdFusion decision.

    Selectable configurations via pipeline_mode parameter or SAFEAD_PIPELINE_MODE env var:
    - 'production' (default): 100% frozen production baseline.
    - 'candidate_b' / 'experimental_alcohol': Baseline + EnhancedAlcoholDetector.
    - 'candidate_c' / 'experimental_violence': Baseline + Safeguarded KineticViolenceDecoupler.
    - 'candidate_d' / 'experimental_all': Baseline + Both Repaired Detectors.
    - 'learned_rf' / 'candidate_e': Learned Random Forest fusion mode.
    - 'learned_rf_hitl' / 'candidate_e_hitl': Learned Random Forest with Policy-First Uncertainty Routing.
    """
    start_total_time = time.time()
    stage_timings = {}
    filename_str = os.path.basename(file_path) if file_path else ""

    # STEP 1: Upload & Input Validation
    step1_start = time.time()
    is_valid, media_type, status_msg = validate_advertisement_file(file_path)
    stage_timings["step1_validation"] = round(time.time() - step1_start, 4)
    if not is_valid:
        return {
            "classification": "UNSAFE_FOR_ALL",
            "display_label": "Unsafe for All",
            "risk_score": 100.0,
            "confidence": 1.0,
            "publication_action": "REJECT",
            "action_badge": "REJECT — UNSAFE FOR ALL",
            "detected_categories": ["Invalid Input"],
            "explanation": f"Input validation failed: {status_msg}",
            "evidence": {},
            "stage_timings": stage_timings,
            "total_processing_time_seconds": round(time.time() - start_total_time, 4)
        }

    print(f"\n[SafeAd AI] Moderate {media_type.upper()}: {filename_str}")
    # STEP 2: Extracting Content (Adaptive Keyframes & Audio)
    step2_start = time.time()
    frames = []
    sampling_meta = {}

    if media_type == "video":
        frames, sampling_meta = extract_adaptive_keyframes(file_path, target_frames=16, return_metadata=True)
        print(f"[SafeAd AI] Adaptive sampling extracted {len(frames)} keyframes. Strategy: {sampling_meta.get('sampling_strategy')}")
    else:
        try:
            if file_path and os.path.exists(file_path):
                pil_img = Image.open(file_path).convert("RGB")
                frames = [pil_img]
            else:
                frames = []
            sampling_meta = {"duration": 0.0, "total_frames": len(frames), "sampling_strategy": "Single Image Input"}
        except Exception:
            frames = []

    stage_timings["step2_extraction"] = round(time.time() - step2_start, 4)

    # STEP 3: Multilingual OCR Analysis
    step3_start = time.time()
    if media_type == "video" and frames:
        ocr_res = ocr_service.extract_from_video_frames(frames)
    else:
        ocr_res = ocr_service.extract_from_image(file_path if (file_path and os.path.exists(file_path)) else (frames[0] if frames else ""))
    stage_timings["step3_ocr"] = round(time.time() - step3_start, 4)


    ocr_text_combined = f"{title or ''} {caption or ''} {ocr_res.get('ocr_text', '')}".strip()
    ocr_text_safety = text_safety_detector.predict_text(ocr_text_combined)

    # STEP 4: Visual Safety & Adult/NSFW Analysis (Llama Guard Vision & Falconsai ViT)
    step4_start = time.time()
    visual_res = visual_detector.analyze_video_frames(frames) if frames else {"unsafe": False, "confidence": 0.0, "detected_categories": []}
    if hasattr(visual_detector, "unload_model"):
        visual_detector.unload_model()
    nsfw_res = nsfw_detector.predict_video_frames(frames, ocr_text=ocr_text_combined) if frames else {"adult_score": 0.0, "adult_content_detected": False}
    if hasattr(nsfw_detector, "unload_model"):
        nsfw_detector.unload_model()
    model_manager.clear_gpu_memory()
    stage_timings["step4_visual_nsfw"] = round(time.time() - step4_start, 4)

    # STEP 5: Video Violence Analysis (VideoMAE)
    step5_start = time.time()
    violence_res = violence_detector.predict_video_frames(frames, ocr_text=ocr_text_combined, filename=filename_str, sampling_meta=sampling_meta) if (media_type == "video" and frames) else {"violence_score": 0.0, "violence_detected": False, "sampled_segments": []}
    if hasattr(violence_detector, "unload_model"):
        violence_detector.unload_model()
    model_manager.clear_gpu_memory()
    stage_timings["step5_violence"] = round(time.time() - step5_start, 4)

    # STEP 6: Audio Speech Transcription & Safety (Whisper)
    step6_start = time.time()
    audio_res = {
        "audio_available": False,
        "transcript": "",
        "audio_safety": {"violence": 0.0, "adult": 0.0, "child_safety": 0.0, "overall_risk": 0.0},
        "status": "not_applicable"
    }

    if media_type == "video":
        audio_res = audio_service.process_video_audio(file_path, text_safety_detector=text_safety_detector)
        if hasattr(audio_service, "unload_model"):
            audio_service.unload_model()
        model_manager.clear_gpu_memory()
    stage_timings["step6_audio"] = round(time.time() - step6_start, 4)

    # STEP 7: Dedicated Advertisement Risk Layer (Scams, Deceptive Marketing, Explicit Content)
    step7_ad_risk_start = time.time()
    ad_risk_output = ad_risk_analyzer.analyze_ad_risks(
        ocr_text=ocr_res.get("ocr_text", ""),
        transcript=audio_res.get("transcript", ""),
        title=title,
        caption=caption,
        text_safety_output=ocr_text_safety
    )

    # Resolve active pipeline configuration mode
    active_mode = (pipeline_mode or os.getenv("SAFEAD_PIPELINE_MODE", "production")).lower().strip()

    # Experimental Detector 1: EnhancedAlcoholDetector (Candidate B / D)
    alcohol_audit = None
    if active_mode in ["candidate_b", "candidate_d", "experimental_alcohol", "experimental_all", "learned_rf", "experimental_learned_rf", "candidate_e", "learned_rf_hitl", "candidate_e_hitl"]:
        alcohol_audit = experimental_alcohol_detector.evaluate_alcohol(
            ocr_text=ocr_text_combined,
            speech_transcript=audio_res.get("transcript", ""),
            visual_objects=ocr_res.get("detected_objects", [])
        )
        if alcohol_audit["alcohol_detected"]:
            ad_risk_output["explicit_content_score"] = max(
                ad_risk_output.get("explicit_content_score", 0.0),
                alcohol_audit["score"]
            )
            ad_risk_output.setdefault("rule_based_evidence", []).extend(alcohol_audit.get("reasons", []))
            if isinstance(ocr_text_safety, dict):
                ocr_text_safety["alcohol"] = alcohol_audit["score"]

    # Experimental Detector 2: Safeguarded KineticViolenceDecoupler (Candidate C / D)
    violence_audit = None
    route_to_hitl = False
    hitl_reasons = []
    if active_mode in ["candidate_c", "candidate_d", "experimental_violence", "experimental_all"] and media_type == "video":
        segments = violence_res.get("sampled_segments", [])
        raw_mot = 25.0 if violence_res.get("violence_score", 0.0) >= 0.20 else 5.0
        vis_viol_flag = "Violence" in visual_res.get("detected_categories", [])
        violence_audit = experimental_violence_decoupler.evaluate_video_action(
            sampled_segments=segments,
            raw_motion_diff=raw_mot,
            text_context=f"{ocr_text_combined} {audio_res.get('transcript', '')}",
            visual_violence_flag=vis_viol_flag
        )
        violence_res["violence_score"] = violence_audit["violence_score"]
        if violence_audit["requires_human_review"]:
            route_to_hitl = True
            hitl_reasons.extend(violence_audit["policy_reasons"])
        elif not violence_audit["violence_detected"] and not vis_viol_flag:
            visual_res["violence"] = min(visual_res.get("violence", 0.0), 0.18)

    stage_timings["step7_ad_risk_analysis"] = round(time.time() - step7_ad_risk_start, 4)

    # STEP 8: Construct Central SafeAdAssessment Matrix
    assessment = SafeAdAssessment(
        visual={
            "violence": 0.90 if "Violence" in visual_res.get("detected_categories", []) else 0.0,
            "adult": nsfw_res.get("adult_score", 0.0),
            "child_safety": 1.0 if "Child Safety Risk" in visual_res.get("detected_categories", []) else 0.0,
            "unsafe": visual_res.get("unsafe", False),
            "evidence": visual_res.get("detected_categories", []),
            "description": visual_res.get("visual_description", "")
        },
        video={
            "violence": max(violence_res.get("violence_score", 0.0), 0.90 if "Violence" in visual_res.get("detected_categories", []) else 0.0),
            "adult": nsfw_res.get("adult_score", 0.0),
            "child_safety": 0.0,
            "sampled_segments": violence_res.get("sampled_segments", [])
        },
        ocr={
            "text": ocr_res.get("ocr_text", ""),
            "violence": ocr_text_safety.get("violent", 0.0),
            "adult": ocr_text_safety.get("sexual", 0.0),
            "child_safety": ocr_text_safety.get("child_safety", 0.0),
            "scam": ad_risk_output.get("scam_score", 0.0),
            "deceptive": ad_risk_output.get("deceptive_marketing_score", 0.0),
            "confidence": ocr_res.get("ocr_confidence", 0.0),
            "evidence": ocr_text_safety.get("detected_categories", [])
        },
        audio={
            "audio_available": audio_res.get("audio_available", False),
            "language": audio_res.get("language", "English"),
            "transcript": audio_res.get("transcript", ""),
            "violence": audio_res.get("audio_safety", {}).get("violence", 0.0),
            "adult": audio_res.get("audio_safety", {}).get("adult", 0.0),
            "child_safety": audio_res.get("audio_safety", {}).get("child_safety", 0.0),
            "scam": ad_risk_output.get("scam_score", 0.0) if audio_res.get("transcript") else 0.0,
            "deceptive": ad_risk_output.get("deceptive_marketing_score", 0.0) if audio_res.get("transcript") else 0.0,
            "evidence": []
        },
        text_safety={
            "categories": ocr_text_safety,
            "overall_risk": ocr_text_safety.get("overall_text_risk", 0.0),
            "evidence": ocr_text_safety.get("detected_categories", [])
        },
        advertisement_risks={
            "scam": ad_risk_output.get("scam_score", 0.0),
            "deceptive_marketing": ad_risk_output.get("deceptive_marketing_score", 0.0),
            "explicit_content": ad_risk_output.get("explicit_content_score", 0.0),
            "evidence": ad_risk_output.get("rule_based_evidence", [])
        }
    )

    # STEP 9: Multimodal Safety Evidence Fusion (SafeAdFusion)
    step9_start = time.time()
    if route_to_hitl:
        # Fail-Safe Asymmetry: Low-confidence/conflicting violence routes to HITL and WITHHOLDS publication
        fusion_output = {
            "classification": "REQUIRES_HUMAN_REVIEW",
            "display_label": "Requires Human Review",
            "risk_score": 60.0,
            "confidence": 0.45,
            "publication_action": "HUMAN_REVIEW",
            "action": "ROUTE_TO_HUMAN_REVIEW",
            "action_badge": "HOLD FOR AUDIT — HUMAN REVIEW",
            "publishable": False,
            "age_restriction": None,
            "policy_stage": "FAIL_SAFE_VIOLENCE_DECOUPLING",
            "policy_reasons": hitl_reasons,
            "explanation": f"Ambiguous detector evidence routed to human safety officer review: {'; '.join(hitl_reasons)}. Publication strictly withheld.",
            "detected_categories": ["Detector Ambiguity / Human Review Required"],
            "evidence": {"policy_reasons": hitl_reasons}
        }
    elif active_mode in ["learned_rf", "experimental_learned_rf", "candidate_e"]:
        fusion_output = experimental_learned_rf.evaluate(assessment, alcohol_audit=alcohol_audit, media_type=media_type, enable_policy_first=False)
    elif active_mode in ["learned_rf_hitl", "candidate_e_hitl"]:
        fusion_output = experimental_learned_rf.evaluate(assessment, alcohol_audit=alcohol_audit, media_type=media_type, enable_policy_first=True)
    else:
        fusion_output = fusion_engine.combine_assessment(assessment)
    stage_timings["step9_fusion"] = round(time.time() - step9_start, 4)

    total_processing_time = round(time.time() - start_total_time, 4)
    stage_timings["step10_decision"] = 0.001
    fusion_output["stage_timings"] = stage_timings
    fusion_output["total_processing_time_seconds"] = total_processing_time
    fusion_output["media_type"] = media_type
    fusion_output["extracted_ocr"] = ocr_res.get("ocr_text", "")
    fusion_output["audio_transcript"] = audio_res.get("transcript", "")
    fusion_output["audio_available"] = audio_res.get("audio_available", False)
    fusion_output["pipeline_mode"] = active_mode
    fusion_output["experimental_audit"] = {
        "alcohol": alcohol_audit,
        "violence": violence_audit
    }

    if "evidence" not in fusion_output or not fusion_output["evidence"]:
        fusion_output["evidence"] = assessment.to_dict()
    if "assessment_matrix" not in fusion_output or not fusion_output["assessment_matrix"]:
        fusion_output["assessment_matrix"] = assessment.to_dict()

    model_manager.clear_gpu_memory()

    print(f"[SafeAd AI] Completed inference in {total_processing_time}s. Classification: {fusion_output['classification']} (Score: {fusion_output['risk_score']}/100, Confidence: {int(fusion_output['confidence']*100)}%)")
    return fusion_output


if __name__ == "__main__":
    print("Testing SafeAd AI pipeline initialization...")
    test_res = run_safead_inference("apples_ad.jpg", "Fresh Apples Offer", "Buy fresh apples online!")
    print(test_res)
