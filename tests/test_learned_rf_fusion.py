import os
import sys
import pytest
from unittest.mock import MagicMock

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JB_DIR = os.path.join(ROOT_DIR, "joseph_safead_core")
B2JB_DIR = os.path.join(ROOT_DIR, "joseph_safead_core", "backend2jb")
for p in [B2JB_DIR, JB_DIR]:
    if p not in sys.path:
        sys.path.append(p)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
else:
    sys.path.remove(ROOT_DIR)
    sys.path.insert(0, ROOT_DIR)

from models.experimental_detectors import LearnedRFFusionClassifier
from models.fusion.safead_assessment import SafeAdAssessment
from backend2jb.ai.pipeline import run_safead_inference

def test_learned_rf_classifier_initialization():
    classifier = LearnedRFFusionClassifier()
    assert classifier.model is not None, "Trained Random Forest estimator should load successfully"
    assert len(classifier.classes) == 4
    assert "UNSAFE_FOR_ALL" in classifier.classes

def test_learned_rf_classifier_prediction_deterministic():
    classifier = LearnedRFFusionClassifier()
    assessment = SafeAdAssessment()
    assessment.visual = {"unsafe": False, "violence": 0.05, "adult": 0.02, "child_safety": 0.0}
    assessment.ocr = {"confidence": 0.85, "scam": 0.0, "deceptive": 0.0, "violence": 0.0, "adult": 0.0}
    assessment.text_safety = {"overall_risk": 0.1, "categories": {}}
    assessment.advertisement_risks = {"scam": 0.0, "deceptive": 0.0, "explicit": 0.0}

    res1 = classifier.evaluate(assessment, media_type="image")
    res2 = classifier.evaluate(assessment, media_type="image")
    assert res1["classification"] == res2["classification"]
    assert res1["confidence"] == res2["confidence"]
    assert res1["probabilities"] == res2["probabilities"]

def test_learned_rf_fail_safe_on_missing_model():
    classifier = LearnedRFFusionClassifier(model_path="non_existent_file.joblib")
    assert classifier.model is None
    assessment = SafeAdAssessment()
    res = classifier.evaluate(assessment)
    assert res["classification"] == "REQUIRES_HUMAN_REVIEW"
    assert res["publishable"] is False
    assert res["action"] == "ROUTE_TO_HUMAN_REVIEW"
    assert "FAIL-SAFE" in res["action_badge"]

def test_learned_rf_fail_safe_on_corrupt_features():
    classifier = LearnedRFFusionClassifier()
    classifier.extract_features = MagicMock(side_effect=RuntimeError("Corrupt feature vector"))
    assessment = SafeAdAssessment()
    res = classifier.evaluate(assessment)
    assert res["classification"] == "REQUIRES_HUMAN_REVIEW"
    assert res["publishable"] is False
    assert res["action"] == "ROUTE_TO_HUMAN_REVIEW"
    assert "FAIL-SAFE" in res["action_badge"]

def test_pipeline_routing_modes():
    # Verify that pipeline runs without error across production, candidate_b, learned_rf, and learned_rf_hitl
    # 1. Production Mode
    prod_res = run_safead_inference(
        file_path="",
        title="Sample Ad",
        caption="Sample ad text",
        pipeline_mode="production"
    )
    assert "classification" in prod_res
    assert prod_res.get("pipeline_mode") == "production"

    # 2. Candidate B Mode
    cb_res = run_safead_inference(
        file_path="",
        title="Sample Ad",
        caption="Sample ad text",
        pipeline_mode="candidate_b"
    )
    assert "classification" in cb_res
    assert cb_res.get("pipeline_mode") == "candidate_b"

    # 3. Learned RF Mode
    rf_res = run_safead_inference(
        file_path="",
        title="Sample Ad",
        caption="Sample ad text",
        pipeline_mode="learned_rf"
    )
    assert "classification" in rf_res
    assert rf_res.get("pipeline_mode") == "learned_rf"
    assert "LEARNED_RANDOM_FOREST_FUSION" in rf_res.get("policy_stage", "")

    # 4. Learned RF HITL Mode
    rf_hitl_res = run_safead_inference(
        file_path="",
        title="Sample Ad",
        caption="Sample ad text",
        pipeline_mode="learned_rf_hitl"
    )
    assert "classification" in rf_hitl_res
    assert rf_hitl_res.get("pipeline_mode") == "learned_rf_hitl"
