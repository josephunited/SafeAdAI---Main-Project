import os
import sys
import pytest
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JB_DIR = os.path.join(ROOT_DIR, "joseph_safead_core")
B2JB_DIR = os.path.join(ROOT_DIR, "joseph_safead_core", "backend2jb")
for p in [B2JB_DIR, JB_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend2jb.ai.pipeline import run_safead_inference
from backend.app.services.ai_client_service import AIClientService
from backend.app.services.policy_service import PolicyService
from backend.app.db.models import SafetyClassification, AgeGroup
from frontend.views.user_workspace import is_ad_allowed_for_age
from models.experimental_detectors import LearnedRFFusionClassifier
from models.fusion.safead_assessment import SafeAdAssessment

def test_ai_client_service_defaults_to_learned_rf():
    """Verify AIClientService uses learned_rf mode by default."""
    res = AIClientService.process_advertisement(
        file_path="",
        title="Test Creative",
        caption="Sample caption"
    )
    assert res.get("pipeline_mode") == "learned_rf"
    assert "LEARNED_RANDOM_FOREST_FUSION" in res.get("policy_stage", "")
    assert res.get("probabilities") is not None

def test_ai_client_service_preserves_production_and_candidate_b():
    """Verify explicit access to production baseline and candidate B."""
    res_prod = AIClientService.process_advertisement(
        file_path="",
        title="Test Creative",
        caption="Sample caption",
        pipeline_mode="production"
    )
    assert res_prod.get("pipeline_mode") == "production"
    assert res_prod.get("policy_stage") in ["AGE_POLICY", "PROHIBITED_POLICY", "STANDARD_POLICY", "SafeAdAssessment"]
    assert res_prod.get("experimental_audit", {}).get("alcohol") is None

    res_cb = AIClientService.process_advertisement(
        file_path="",
        title="Budweiser King of Beers",
        caption="Cold draft lager beer",
        pipeline_mode="candidate_b"
    )
    assert res_cb.get("pipeline_mode") == "candidate_b"
    assert res_cb.get("experimental_audit", {}).get("alcohol") is not None

def test_learned_rf_hitl_uncertainty_routing_to_human_review():
    """Verify that when P(UNSAFE_FOR_ALL) >= 0.20 and pred != UFA, it routes to REQUIRES_HUMAN_REVIEW."""
    classifier = LearnedRFFusionClassifier()
    # Mock model.predict_proba to return elevated prohibited risk: P(S18P)=0.45, P(UFA)=0.25, P(SFA)=0.15, P(S14P)=0.15
    # Classes: ['SAFE_14_PLUS', 'SAFE_18_PLUS', 'SAFE_FOR_ALL', 'UNSAFE_FOR_ALL']
    mock_probs = [0.15, 0.45, 0.15, 0.25]
    classifier.model = MagicMock()
    classifier.model.predict_proba.return_value = [mock_probs]
    
    assessment = SafeAdAssessment()
    # Standard RF: predicts SAFE_18_PLUS
    res_standard = classifier.evaluate(assessment, enable_policy_first=False)
    assert res_standard["classification"] == "SAFE_18_PLUS"
    assert res_standard["publishable"] is True
    
    # Safety-Prioritized HITL: routes to REQUIRES_HUMAN_REVIEW
    res_hitl = classifier.evaluate(assessment, enable_policy_first=True)
    assert res_hitl["classification"] == "REQUIRES_HUMAN_REVIEW"
    assert res_hitl["publishable"] is False
    assert res_hitl["publication_action"] == "HUMAN_REVIEW"
    assert "HOLD FOR AUDIT" in res_hitl["action_badge"]
    assert res_hitl["policy_stage"] == "POLICY_FIRST_UNCERTAINTY_GATE"

def test_learned_rf_fail_safe_on_model_or_feature_failure():
    """Verify strict fail-closed safety invariant: never auto-approve on failure."""
    # 1. Missing model
    classifier = LearnedRFFusionClassifier(model_path="non_existent_file.joblib")
    assessment = SafeAdAssessment()
    res_missing = classifier.evaluate(assessment)
    assert res_missing["classification"] == "REQUIRES_HUMAN_REVIEW"
    assert res_missing["publishable"] is False
    assert "FAIL-SAFE" in res_missing["action_badge"]

    # 2. Corrupt features exception
    classifier_loaded = LearnedRFFusionClassifier()
    classifier_loaded.extract_features = MagicMock(side_effect=RuntimeError("Feature extraction crashed"))
    res_corrupt = classifier_loaded.evaluate(assessment)
    assert res_corrupt["classification"] == "REQUIRES_HUMAN_REVIEW"
    assert res_corrupt["publishable"] is False
    assert "FAIL-SAFE" in res_corrupt["action_badge"]

def test_frontend_feed_age_gating_rules():
    """Verify that age-gating rules in user workspace remain 100% effective."""
    # 1. REQUIRES_HUMAN_REVIEW: blocked from all age groups
    assert is_ad_allowed_for_age("REQUIRES_HUMAN_REVIEW", "Less than 14") is False
    assert is_ad_allowed_for_age("REQUIRES_HUMAN_REVIEW", "14 to 17") is False
    assert is_ad_allowed_for_age("REQUIRES_HUMAN_REVIEW", "18 and above") is False

    # 2. UNSAFE_FOR_ALL: blocked from all age groups
    assert is_ad_allowed_for_age("UNSAFE_FOR_ALL", "Less than 14") is False
    assert is_ad_allowed_for_age("UNSAFE_FOR_ALL", "14 to 17") is False
    assert is_ad_allowed_for_age("UNSAFE_FOR_ALL", "18 and above") is False

    # 3. SAFE_18_PLUS: blocked from <14 and 14-17; allowed ONLY for 18+
    assert is_ad_allowed_for_age("SAFE_18_PLUS", "Less than 14") is False
    assert is_ad_allowed_for_age("SAFE_18_PLUS", "14 to 17") is False
    assert is_ad_allowed_for_age("SAFE_18_PLUS", "18 and above") is True

    # 4. SAFE_14_PLUS: blocked from <14; allowed for 14-17 and 18+
    assert is_ad_allowed_for_age("SAFE_14_PLUS", "Less than 14") is False
    assert is_ad_allowed_for_age("SAFE_14_PLUS", "14 to 17") is True
    assert is_ad_allowed_for_age("SAFE_14_PLUS", "18 and above") is True

    # 5. SAFE_FOR_ALL: allowed for all age groups
    assert is_ad_allowed_for_age("SAFE_FOR_ALL", "Less than 14") is True
    assert is_ad_allowed_for_age("SAFE_FOR_ALL", "14 to 17") is True
    assert is_ad_allowed_for_age("SAFE_FOR_ALL", "18 and above") is True
