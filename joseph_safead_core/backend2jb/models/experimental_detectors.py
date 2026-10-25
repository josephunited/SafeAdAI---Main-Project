"""
SafeAd AI (SAFE-VISION) — Experimental Repaired Detectors.
Contains experimental detector modules approved for controlled testing:
1. EnhancedAlcoholDetector: Gated keyword, brand, and visual token alcohol detector with strict negative controls.
2. KineticViolenceDecoupler: Video action recognizer with fail-safe human review routing.

IMPORTANT:
- These modules are experimental and MUST NOT be active in the production pipeline by default.
- Production scoring formulas in models/fusion/safead_fusion.py remain strictly frozen.
"""

import os
import re
from typing import Dict, Any, List, Optional


class EnhancedAlcoholDetector:
    """
    Experimental detector for alcoholic beverages and drinking establishments.
    Adheres to strict negative-control principles:
    - Generic bottle / glass presence alone NEVER equates to alcohol.
    - Explicit non-alcoholic claims ('0.0%', 'non-alcoholic', 'virgin') force suppression.
    - Requires verified brand or explicit ABV keyword before triggering.
    """
    VERIFIED_ALCOHOL_BRANDS = [
        "budweiser", "heineken", "corona", "guinness", "stella artois",
        "carlsberg", "smirnoff", "absolut", "bacardi", "captain morgan",
        "jack daniel's", "jack daniels", "johnnie walker", "chivas regal",
        "glenfiddich", "jameson", "hennessy", "remy martin", "grey goose",
        "tanqueray", "bombay sapphire", "gordon's", "baileys", "kahlua",
        "moet", "dom perignon", "veuve clicquot", "tuborg", "kingfisher"
    ]

    EXPLICIT_ALCOHOL_KEYWORDS = [
        "whiskey", "whisky", "vodka", "rum", "tequila", "gin", "brandy",
        "cognac", "beer", "lager", "ale", "stout", "wine", "champagne",
        "liqueur", "liquor", "cocktail", "bourbon", "scotch", "draught beer",
        "hard seltzer", "alcoholic beverage", "craft beer", "brewery", "distillery"
    ]

    GENERIC_CONTAINERS = [
        "bottle", "glass", "wine glass", "beer glass", "cocktail glass",
        "tumbler", "drink", "liquid", "cup", "mug", "bar", "restaurant"
    ]

    EXPLICIT_NON_ALCOHOLIC_TOKENS = [
        "0.0%", "0.0 %", "non-alcoholic", "non alcoholic", "alcohol-free",
        "alcohol free", "zero alcohol", "virgin cocktail", "virgin mojito",
        "soft drink", "fruit juice", "sparkling water", "energy drink",
        "olive oil", "cooking oil", "vinegar"
    ]

    def evaluate_alcohol(
        self,
        ocr_text: str = "",
        speech_transcript: str = "",
        visual_objects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates advertisement text, transcript, and visual object tokens for alcohol promotion.
        """
        vis_objs = visual_objects or []
        combined_text = f"{ocr_text} {speech_transcript}".lower()
        vis_tokens_lower = [str(o).lower() for o in vis_objs]

        # 1. Negative Control Check: Explicit non-alcoholic claims
        has_non_alc = any(t in combined_text for t in self.EXPLICIT_NON_ALCOHOLIC_TOKENS)
        if has_non_alc:
            return {
                "alcohol_detected": False,
                "confidence": 0.95,
                "score": 0.0,
                "reasons": ["Explicit non-alcoholic / zero-alcohol product declaration detected."],
                "matched_brands": [],
                "matched_keywords": []
            }

        matched_brands = [b for b in self.VERIFIED_ALCOHOL_BRANDS if re.search(r'\b' + re.escape(b) + r'\b', combined_text)]
        matched_keywords = [k for k in self.EXPLICIT_ALCOHOL_KEYWORDS if re.search(r'\b' + re.escape(k) + r'\b', combined_text)]

        has_container = any(c in vis_tokens_lower or c in combined_text for c in self.GENERIC_CONTAINERS)

        # 2. Strict Negative Control: Container presence alone NEVER equates to alcohol
        if not matched_brands and not matched_keywords:
            return {
                "alcohol_detected": False,
                "confidence": 0.90,
                "score": 0.0,
                "reasons": ["No verified alcohol brands or keywords found. Generic containers decoupled."],
                "matched_brands": [],
                "matched_keywords": []
            }

        # 3. Verified Detection
        if matched_brands:
            score = 0.95 if has_container else 0.85
            reasons = [f"Verified alcohol brand identified: {', '.join(matched_brands)}"]
            if has_container:
                reasons.append("Alcoholic beverage container/establishment context confirmed.")
            return {
                "alcohol_detected": True,
                "confidence": score,
                "score": score,
                "reasons": reasons,
                "matched_brands": matched_brands,
                "matched_keywords": matched_keywords
            }

        if matched_keywords:
            score = 0.85 if has_container else 0.70
            reasons = [f"Explicit alcohol category keyword identified: {', '.join(matched_keywords)}"]
            return {
                "alcohol_detected": True,
                "confidence": score,
                "score": score,
                "reasons": reasons,
                "matched_brands": [],
                "matched_keywords": matched_keywords
            }

        return {
            "alcohol_detected": False,
            "confidence": 0.90,
            "score": 0.0,
            "reasons": [],
            "matched_brands": [],
            "matched_keywords": []
        }


class KineticViolenceDecoupler:
    """
    Experimental decoupler separating genuine interpersonal violence from
    camera motion, athletic activities, and dynamic videography.

    Fail-Safe Routing Principle:
    - If visual safety flags violence and VideoMAE confidence is low (< 0.30)
      without a high-confidence benign athletic explanation, it ROUTES TO HUMAN REVIEW
      and WITHHOLDS publication.
    """
    INTERPERSONAL_VIOLENCE_ACTIONS = [
        "punching person", "slapping", "kicking", "wrestling", "brawling",
        "fighting", "sword fighting", "shooting gun", "stabbing", "headbutting"
    ]

    BENIGN_ATHLETIC_AND_MOTION_ACTIONS = [
        "running on treadmill", "jogging", "marathon running", "swimming",
        "skateboarding", "snowboarding", "gymnastics tumbling", "dancing",
        "playing basketball", "playing tennis", "driving car", "cycling",
        "parkour", "pole vault", "cheerleading"
    ]

    def evaluate_video_action(
        self,
        sampled_segments: List[Dict[str, Any]],
        raw_motion_diff: float = 0.0,
        text_context: str = "",
        visual_violence_flag: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates sampled video action segments with fail-safe ambiguity routing.
        """
        top_violence_prob = 0.0
        violent_label = None
        top_benign_prob = 0.0
        benign_label = None

        for seg in sampled_segments:
            lbl = str(seg.get("label", "")).lower()
            score = float(seg.get("score", 0.0))

            if any(v_act in lbl for v_act in self.INTERPERSONAL_VIOLENCE_ACTIONS):
                if score > top_violence_prob:
                    top_violence_prob = score
                    violent_label = lbl

            if any(a_act in lbl for a_act in self.BENIGN_ATHLETIC_AND_MOTION_ACTIONS):
                if score > top_benign_prob:
                    top_benign_prob = score
                    benign_label = lbl

        text_lower = text_context.lower()
        has_violent_text = any(w in text_lower for w in ["fight", "murder", "blood", "gun", "knife", "combat", "war", "assault", "brawl"])

        # Case 1: Confirmed Interpersonal Violence
        if top_violence_prob >= 0.40 or (top_violence_prob >= 0.25 and has_violent_text) or has_violent_text:
            return {
                "violence_detected": True,
                "requires_human_review": False,
                "action": "REJECT",
                "violence_score": round(max(top_violence_prob, 0.85 if has_violent_text else 0.70), 4),
                "top_violent_action": violent_label,
                "policy_reasons": [f"Prohibited interpersonal violence detected ({violent_label or 'text cues'})"],
                "explanation": "Severe physical aggression or violent weapon action confirmed."
            }

        # Case 2: Visual Violence Flagged with Low-Confidence / Conflicting VideoMAE Evidence (FAIL-SAFE)
        if visual_violence_flag:
            if top_benign_prob >= 0.45 and not has_violent_text:
                return {
                    "violence_detected": False,
                    "requires_human_review": False,
                    "action": "APPROVE",
                    "violence_score": 0.12,
                    "policy_reasons": [f"Visual motion explained by high-confidence athletic action ({benign_label})"],
                    "explanation": "High-confidence athletic action confirmed; camera motion decoupled."
                }
            else:
                # VideoMAE returned diffuse or low confidence (< 0.30) -> DO NOT SUPPRESS. Route to HITL!
                return {
                    "violence_detected": False,
                    "requires_human_review": True,
                    "action": "ROUTE_TO_HUMAN_REVIEW",
                    "violence_score": 0.60,
                    "policy_reasons": ["Detector conflict: Visual guard flagged violence, but video classifier returned low confidence (< 0.30). Withheld for human safety officer review."],
                    "explanation": "Ambiguous/conflicting video violence evidence. Publication withheld pending human review."
                }

        # Case 3: Visual Violence NOT Flagged
        if top_benign_prob >= 0.15:
            return {
                "violence_detected": False,
                "requires_human_review": False,
                "action": "APPROVE",
                "violence_score": 0.05,
                "policy_reasons": [],
                "explanation": f"Benign motion identified ({benign_label}); camera movement decoupled."
            }

        final_v = min(0.18, raw_motion_diff / 100.0)
        return {
            "violence_detected": False,
            "requires_human_review": False,
            "action": "APPROVE",
            "violence_score": final_v,
            "policy_reasons": [],
            "explanation": "Standard commercial video pacing; no violence detected."
        }


class LearnedRFFusionClassifier:
    """
    Experimental learned fusion classifier using the Phase 8E Random Forest artifact.
    Strictly preserves fail-closed safety:
    - If model file is missing or unreadable, fails closed to REQUIRES_HUMAN_REVIEW (publishable=False).
    - If feature extraction encounters an exception, fails closed to REQUIRES_HUMAN_REVIEW (publishable=False).
    - Never automatically approves an advertisement on failure.
    - Optional policy-first uncertainty gate routes borderline prohibited cases (P(UNSAFE_FOR_ALL) >= 0.20) to human review.
    """
    FEATURE_NAMES = [
        "is_video", "visual_unsafe", "visual_violence", "visual_adult", "visual_child_safety",
        "video_violence", "video_adult", "video_child_safety",
        "ocr_confidence", "ocr_violence", "ocr_adult", "ocr_child_safety", "ocr_scam", "ocr_deceptive",
        "has_audio", "audio_violence", "audio_adult", "audio_child_safety", "audio_scam", "audio_deceptive",
        "text_overall_risk", "text_violent", "text_sexual", "text_hate", "text_gambling", "text_alcohol", "text_drugs", "text_misleading",
        "ad_risk_scam", "ad_risk_deceptive", "ad_risk_explicit",
        "alcohol_detected", "alcohol_score"
    ]

    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model_path = model_path
        else:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            self.model_path = os.path.join(base_dir, "results", "phase8_learned_fusion", "final_random_forest_estimator.joblib")
        self.artifact = None
        self.model = None
        self.classes = None
        self._load_model()

    def _load_model(self):
        try:
            import joblib
            if os.path.exists(self.model_path):
                self.artifact = joblib.load(self.model_path)
                self.model = self.artifact.get("estimator")
                self.classes = list(self.artifact.get("classes", []))
        except Exception:
            self.model = None
            self.classes = None

    def extract_features(
        self,
        assessment: Any,
        alcohol_audit: Optional[Dict[str, Any]] = None,
        media_type: str = "image"
    ) -> List[float]:
        data = assessment.to_dict() if hasattr(assessment, "to_dict") else (assessment or {})
        vis = data.get("visual", {})
        vid = data.get("video", {})
        ocr = data.get("ocr", {})
        aud = data.get("audio", {})
        ts = data.get("text_safety", {})
        ts_cats = ts.get("categories", {}) if isinstance(ts.get("categories"), dict) else {}
        ad_risks = data.get("advertisement_risks", {})
        alc = alcohol_audit or {}

        return [
            1.0 if media_type == "video" else 0.0,
            1.0 if vis.get("unsafe") else 0.0,
            float(vis.get("violence", 0.0) or 0.0),
            float(vis.get("adult", 0.0) or 0.0),
            float(vis.get("child_safety", 0.0) or 0.0),
            float(vid.get("violence", 0.0) or 0.0),
            float(vid.get("adult", 0.0) or 0.0),
            float(vid.get("child_safety", 0.0) or 0.0),
            float(ocr.get("confidence", 0.0) or 0.0),
            float(ocr.get("violence", 0.0) or 0.0),
            float(ocr.get("adult", 0.0) or 0.0),
            float(ocr.get("child_safety", 0.0) or 0.0),
            float(ocr.get("scam", 0.0) or 0.0),
            float(ocr.get("deceptive", 0.0) or 0.0),
            1.0 if aud.get("audio_available") else 0.0,
            float(aud.get("violence", 0.0) or 0.0),
            float(aud.get("adult", 0.0) or 0.0),
            float(aud.get("child_safety", 0.0) or 0.0),
            float(aud.get("scam", 0.0) or 0.0),
            float(aud.get("deceptive", 0.0) or 0.0),
            float(ts.get("overall_risk", 0.0) or 0.0),
            float(ts_cats.get("violent", 0.0) or 0.0),
            float(ts_cats.get("sexual", 0.0) or 0.0),
            float(ts_cats.get("hate", 0.0) or 0.0),
            float(ts_cats.get("gambling", 0.0) or 0.0),
            float(ts_cats.get("alcohol", 0.0) or 0.0),
            float(ts_cats.get("drugs", 0.0) or 0.0),
            float(ts_cats.get("misleading", 0.0) or 0.0),
            float(ad_risks.get("scam", 0.0) or 0.0),
            float(ad_risks.get("deceptive_marketing", 0.0) or 0.0),
            float(ad_risks.get("explicit_content", 0.0) or 0.0),
            1.0 if alc.get("alcohol_detected") else 0.0,
            float(alc.get("score", 0.0) or 0.0),
        ]

    def build_multimodal_explanation(
        self,
        assessment: Any,
        pred_class: str,
        display: str,
        confidence_pct: int,
        media_type: str = "video"
    ) -> str:
        """
        Synthesizes an exhaustive multimodal explanation incorporating visual scene understanding,
        kinetic video actions, overlaid OCR text, speech dialogue, and policy rationale.
        """
        lines = [
            f"Learned multimodal Random Forest fusion classified advertisement as {display} ({pred_class}) with {confidence_pct}% confidence."
        ]

        details = []

        # 1. Visual Scene & Imagery
        vis = getattr(assessment, "visual", {}) or {}
        vis_desc = vis.get("description", "")
        vis_cats = vis.get("evidence", []) or []
        vis_adult = float(vis.get("adult", 0.0) or 0.0)
        vis_viol = float(vis.get("violence", 0.0) or 0.0)

        vis_parts = []
        if vis_desc:
            vis_parts.append(f'observed scenes: "{vis_desc}"')
        if vis_cats:
            vis_parts.append(f"visual safety flags: {', '.join(vis_cats)}")
        if vis_viol >= 0.30:
            vis_parts.append(f"visual violence score: {vis_viol:.2f}")
        if vis_adult >= 0.35:
            vis_parts.append(f"adult/NSFW score: {vis_adult:.2f} (suggestive imagery / skin ratio detected)")
        elif vis_adult > 0.10:
            vis_parts.append(f"mild adult score: {vis_adult:.2f}")

        if vis_parts:
            details.append(f"- **Visual Scene Analysis**: {'; '.join(vis_parts)}.")
        else:
            details.append("- **Visual Scene Analysis**: Standard commercial creative; no visual policy violations detected.")

        # 2. Kinetic Video Action & Motion (for video)
        if media_type == "video":
            vid = getattr(assessment, "video", {}) or {}
            vid_viol = float(vid.get("violence", 0.0) or 0.0)
            segments = vid.get("sampled_segments", []) or []

            action_strs = []
            for s in segments[:4]:
                lbl = s.get("label", "")
                sc = float(s.get("score", 0.0) or 0.0)
                if sc > 0.10:
                    action_strs.append(f"{lbl} ({int(sc*100)}%)")

            vid_parts = []
            if action_strs:
                vid_parts.append(f"kinetic actions identified: {', '.join(action_strs)}")
            if vid_viol >= 0.25:
                vid_parts.append(f"physical violence score: {vid_viol:.2f}")

            if vid_parts:
                details.append(f"- **Video Action & Motion**: {'; '.join(vid_parts)}.")
            else:
                details.append("- **Video Action & Motion**: Routine camera movement; no kinetic combat or physical aggression detected.")

        # 3. On-Screen Text (OCR)
        ocr = getattr(assessment, "ocr", {}) or {}
        ocr_tx = str(ocr.get("text", "") or "").strip()
        ocr_ev = ocr.get("evidence", []) or []
        if ocr_tx:
            sample_tx = ocr_tx[:140] + ("..." if len(ocr_tx) > 140 else "")
            ev_str = f" [Flags: {', '.join(ocr_ev)}]" if ocr_ev else ""
            details.append(f'- **On-Screen Text (OCR)**: Overlay text detected: *"{sample_tx}"*{ev_str}.')
        else:
            details.append("- **On-Screen Text (OCR)**: No prominent overlaid text or prohibited promotional slogans detected.")

        # 4. Audio Speech & Dialogue (Whisper)
        aud = getattr(assessment, "audio", {}) or {}
        transcript = str(aud.get("transcript", "") or "").strip()
        has_audio = aud.get("audio_available", False)
        if transcript:
            sample_tr = transcript[:140] + ("..." if len(transcript) > 140 else "")
            details.append(f'- **Audio Speech & Dialogue**: Spoken dialogue transcribed: *"{sample_tr}"*.')
        elif has_audio:
            details.append("- **Audio Speech & Dialogue**: Background audio/soundtrack detected with no prohibited spoken dialogue.")
        else:
            details.append("- **Audio Speech & Dialogue**: No audio track present in media file.")

        # 5. Policy Enforcement Rationale
        if pred_class == "UNSAFE_FOR_ALL":
            details.append("- **Policy Enforcement**: Prohibited from publication across all feeds due to severe content violations (violence, explicit content, or safety policy breach).")
        elif pred_class == "SAFE_18_PLUS":
            details.append("- **Policy Enforcement**: Restricted to adult audiences (18+) due to mature themes, adult content, or age-gated commercial material.")
        elif pred_class == "SAFE_14_PLUS":
            details.append("- **Policy Enforcement**: Age-gated to teen audiences (14+) for mild thematic elements; prohibited for younger children.")
        elif pred_class == "REQUIRES_HUMAN_REVIEW":
            details.append("- **Policy Enforcement**: Withheld from automated publication; routed to human safety officer queue for manual review.")
        else:
            details.append("- **Policy Enforcement**: Approved for unrestricted publication across all user age tiers.")

        if details:
            lines.append("\n" + "\n".join(details))

        return "\n".join(lines)

    def evaluate(
        self,
        assessment: Any,
        alcohol_audit: Optional[Dict[str, Any]] = None,
        media_type: str = "image",
        enable_policy_first: bool = False
    ) -> Dict[str, Any]:
        # Fail-Safe Gate 1: Missing or unloaded model fails closed to Human Review
        if self.model is None or not self.classes:
            return {
                "classification": "REQUIRES_HUMAN_REVIEW",
                "display_label": "Requires Human Review",
                "publication_action": "HUMAN_REVIEW",
                "action": "ROUTE_TO_HUMAN_REVIEW",
                "action_badge": "FAIL-SAFE — MODERATOR REVIEW QUEUE",
                "publishable": False,
                "risk_score": 65.0,
                "confidence": 0.40,
                "policy_stage": "LEARNED_FUSION_FAIL_SAFE",
                "policy_reasons": ["Learned fusion model artifact unavailable; failed closed to human review for safety compliance."],
                "explanation": "Learned fusion model failed to load. Publication strictly withheld pending moderator review.",
                "probabilities": {}
            }

        try:
            import numpy as np
            features = self.extract_features(assessment, alcohol_audit, media_type)
            if len(features) != len(self.FEATURE_NAMES):
                raise ValueError(f"Feature count mismatch: expected {len(self.FEATURE_NAMES)}, got {len(features)}")

            X = np.array([features], dtype=np.float32)
            probs = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = str(self.classes[pred_idx])
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(self.classes, probs)}

            # Policy-first uncertainty guard: if P(UNSAFE_FOR_ALL) >= 0.20 and not predicted UFA, route to HITL
            p_ufa = prob_dict.get("UNSAFE_FOR_ALL", 0.0)
            if enable_policy_first and p_ufa >= 0.20 and pred_class != "UNSAFE_FOR_ALL":
                return {
                    "classification": "REQUIRES_HUMAN_REVIEW",
                    "display_label": "Requires Human Review",
                    "publication_action": "HUMAN_REVIEW",
                    "action": "ROUTE_TO_HUMAN_REVIEW",
                    "action_badge": "HOLD FOR AUDIT — HUMAN REVIEW",
                    "publishable": False,
                    "age_restriction": None,
                    "risk_score": 60.0,
                    "confidence": round(float(np.max(probs)), 2),
                    "policy_stage": "POLICY_FIRST_UNCERTAINTY_GATE",
                    "policy_reasons": [f"Elevated prohibited risk probability ({p_ufa*100:.1f}%) detected in learned fusion."],
                    "explanation": f"Learned fusion flagged elevated prohibited probability ({p_ufa*100:.1f}%). Withheld for human moderator review.",
                    "probabilities": prob_dict
                }

            # Standard four-class decisions
            if pred_class == "UNSAFE_FOR_ALL":
                action = "REJECT"
                badge = "REJECT — UNSAFE FOR ALL"
                publishable = False
                display = "Unsafe for All"
                age_restriction = None
                risk_score = round(max(70.0, p_ufa * 100.0), 1)
            elif pred_class == "SAFE_18_PLUS":
                action = "AGE_RESTRICT"
                badge = "AGE RESTRICTED — 18+"
                publishable = True
                display = "18+"
                age_restriction = "18+"
                risk_score = round(max(45.0, prob_dict.get("SAFE_18_PLUS", 0.45) * 70.0), 1)
            elif pred_class == "SAFE_14_PLUS":
                action = "AGE_RESTRICT"
                badge = "AGE RESTRICTED — 14+"
                publishable = True
                display = "14+"
                age_restriction = "14+"
                risk_score = round(max(25.0, prob_dict.get("SAFE_14_PLUS", 0.25) * 50.0), 1)
            else:
                action = "APPROVE"
                badge = "APPROVED — SAFE FOR ALL"
                publishable = True
                display = "Safe for All"
                age_restriction = None
                risk_score = round(min(24.9, (1.0 - probs[pred_idx]) * 35.0), 1)

            detailed_explanation = self.build_multimodal_explanation(
                assessment=assessment,
                pred_class=pred_class,
                display=display,
                confidence_pct=int(np.max(probs)*100),
                media_type=media_type
            )

            return {
                "classification": pred_class,
                "display_label": display,
                "publication_action": action,
                "action_badge": badge,
                "publishable": publishable,
                "age_restriction": age_restriction,
                "risk_score": risk_score,
                "confidence": round(float(np.max(probs)), 2),
                "policy_stage": "LEARNED_RANDOM_FOREST_FUSION",
                "policy_reasons": [f"Learned Random Forest classified as {pred_class} (Confidence: {int(np.max(probs)*100)}%)"],
                "explanation": detailed_explanation,
                "probabilities": prob_dict
            }
        except Exception as e:
            # Fail-Safe Gate 2: Any feature extraction or runtime error strictly fails closed to Human Review
            return {
                "classification": "REQUIRES_HUMAN_REVIEW",
                "display_label": "Requires Human Review",
                "publication_action": "HUMAN_REVIEW",
                "action": "ROUTE_TO_HUMAN_REVIEW",
                "action_badge": "FAIL-SAFE — MODERATOR REVIEW QUEUE",
                "publishable": False,
                "risk_score": 65.0,
                "confidence": 0.40,
                "policy_stage": "LEARNED_FUSION_EXCEPTION_FAIL_SAFE",
                "policy_reasons": [f"Learned fusion runtime error ({str(e)}); failed closed to human review."],
                "explanation": f"Learned fusion encountered an error: {str(e)}. Publication strictly withheld.",
                "probabilities": {}
            }

