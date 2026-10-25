import os
import json
import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
oof_path = os.path.join(ROOT_DIR, "results", "phase8_learned_fusion", "out_of_fold_predictions.csv")
baseline_path = os.path.join(ROOT_DIR, "results", "real_pipeline_baseline", "saved_detector_outputs.json")
doc_path = os.path.join(ROOT_DIR, "docs", "FINAL_VERIFIED_MODEL_COMPARISON.md")

@pytest.fixture(scope="module")
def loaded_data():
    df = pd.read_csv(oof_path)
    with open(baseline_path, "r") as f:
        saved = json.load(f)
    baseline_preds = {k: v["predicted_label"] for k, v in saved.items()}
    df["baseline_prediction"] = df["sample_id"].map(baseline_preds)
    classes = ["SAFE_FOR_ALL", "SAFE_14_PLUS", "SAFE_18_PLUS", "UNSAFE_FOR_ALL"]
    return df, classes

def test_authoritative_sample_counts_and_supports(loaded_data):
    df, classes = loaded_data
    assert len(df) == 194, "Total development samples must be exactly 194"
    supports = df["ground_truth"].value_counts().to_dict()
    assert supports == {"SAFE_FOR_ALL": 75, "SAFE_14_PLUS": 75, "SAFE_18_PLUS": 34, "UNSAFE_FOR_ALL": 10}

@pytest.mark.parametrize("model_col,expected_diag,expected_acc_pct", [
    ("baseline_prediction", [48, 3, 9, 2], 31.96),
    ("candidate_b_prediction", [48, 3, 12, 2], 33.51),
    ("logistic_regression_oof_prediction", [21, 34, 17, 2], 38.14),
    ("random_forest_oof_prediction", [33, 31, 18, 4], 44.33),
])
def test_non_abstaining_models_diagonal_and_accuracy_assertions(loaded_data, model_col, expected_diag, expected_acc_pct):
    df, classes = loaded_data
    y_true = df["ground_truth"]
    y_pred = df[model_col]

    # Confusion matrix
    cm = pd.crosstab(y_true, y_pred).reindex(index=classes, columns=classes, fill_value=0)
    
    # 1. Matrix sum = 194
    assert cm.values.sum() == 194, f"{model_col}: Confusion matrix sum must equal 194"

    # 2. Diagonal sum = number of correct predictions
    diag = np.diag(cm.values).tolist()
    assert diag == expected_diag, f"{model_col}: Diagonal {diag} does not match expected {expected_diag}"
    
    correct_count = (y_true == y_pred).sum()
    assert sum(diag) == correct_count, f"{model_col}: Diagonal sum {sum(diag)} != correct count {correct_count}"

    # 3. Class-wise recall numerators equal diagonal cells
    for i, c in enumerate(classes):
        assert int(cm.loc[c, c]) == diag[i]

    # 4. Accuracy = diagonal sum / 194
    acc = sum(diag) / 194
    assert round(acc * 100, 2) == expected_acc_pct, f"{model_col}: Accuracy {acc*100:.2f}% != {expected_acc_pct}%"

def test_rf_hitl_abstention_and_selective_accuracy(loaded_data):
    df, classes = loaded_data
    y_true = df["ground_truth"]
    y_pred = df["random_forest_hitl_prediction"]

    # 5-column confusion matrix
    hitl_cols = classes + ["REQUIRES_HUMAN_REVIEW"]
    cm = pd.crosstab(y_true, y_pred).reindex(index=classes, columns=hitl_cols, fill_value=0)
    assert cm.values.sum() == 194

    abstained = (y_pred == "REQUIRES_HUMAN_REVIEW").sum()
    automated = 194 - abstained
    assert abstained == 49, "RF+HITL must abstain exactly 49 samples"
    assert automated == 145, "RF+HITL automated decisions must be exactly 145"

    # Automated diagonal
    diag = [int(cm.loc[c, c]) for c in classes]
    assert diag == [29, 14, 15, 4], f"HITL automated diagonal mismatch: {diag}"
    hitl_correct = sum(diag)
    assert hitl_correct == 62, "Automated 4-class correct count must equal 62"

    # Invariant: human review withholding is NEVER counted as correct 4-class classification
    raw_acc = hitl_correct / 194
    assert round(raw_acc * 100, 2) == 31.96

    # Selective accuracy on automated coverage
    selective_acc = hitl_correct / 145
    assert round(selective_acc * 100, 2) == 42.76

def test_document_artifact_consistency():
    assert os.path.exists(doc_path), "FINAL_VERIFIED_MODEL_COMPARISON.md must exist"
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Assert key audited values are present and exact
    assert "31.96% (62/194)" in content
    assert "33.51% (65/194)" in content
    assert "38.14% (74/194)" in content
    assert "44.33% (86/194)" in content
    assert "42.76% (62/145)" in content
    assert "40.00% (4/10)" in content
    assert "17.65% (6/34)" in content
    assert "6.00% (9/150)" in content
