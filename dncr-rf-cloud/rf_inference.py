"""Portable inference for the saved five-member random forest model."""
from __future__ import annotations

import json
from math import factorial
from pathlib import Path
import numpy as np

FEATURE_LABELS = {
    "Age": "Age",
    "MOCA_Score": "MOCA Score",
    "Depression": "Depression",
    "GFR": "GFR",
    "Operation_Time": "Operation Time",
}


def load_model(name: str = "RF", directory: Path | None = None) -> dict:
    if name != "RF":
        raise ValueError("This app uses the RF model.")
    directory = directory or Path(__file__).resolve().parent / "models"
    model = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if model["schema_version"] != 2 or model["model"] != name or len(model["members"]) != 5:
        raise ValueError("The model file is incomplete or inconsistent.")
    for member in model["members"]:
        for key in ("root_nodes", "children_left", "children_right", "feature"):
            member[key] = np.asarray(member[key], dtype=np.int64)
        for key in ("threshold", "positive_probability"):
            member[key] = np.asarray(member[key], dtype=np.float64)
    return model


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def forest_probability(member: dict, values: np.ndarray) -> np.ndarray:
    """Traverse the exported forest, preserving scikit-learn's float32 input."""
    nodes = np.broadcast_to(member["root_nodes"], (len(values), len(member["root_nodes"]))).copy()
    for _ in range(member["max_depth"] + 1):
        active = member["children_left"][nodes] != -1
        if not active.any():
            # Sum in estimator order, matching the original forest predict_proba.
            leaves = member["positive_probability"][nodes]
            total = np.zeros(len(values), dtype=np.float64)
            for tree_index in range(leaves.shape[1]):
                total += leaves[:, tree_index]
            return total / leaves.shape[1]
        rows, trees = np.nonzero(active)
        current = nodes[rows, trees]
        go_left = values[rows, member["feature"][current]] <= member["threshold"][current]
        nodes[rows, trees] = np.where(go_left, member["children_left"][current], member["children_right"][current])
    raise ValueError("The RF model contains an inconsistent tree structure.")


def predict_array(model: dict, values) -> np.ndarray:
    """Evaluate complete inputs; fractional baseline values are allowed for SHAP."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != len(model["features"]) or not np.isfinite(values).all():
        raise ValueError("Five finite predictor values are required.")
    clipped = np.clip(values, model["clip_lower"], model["clip_upper"])
    tree_input = clipped.astype(np.float32)
    probabilities = []
    for member in model["members"]:
        probabilities.append(forest_probability(member, tree_input))
    average = np.clip(np.mean(probabilities, axis=0), 1e-6, 1 - 1e-6)
    log_odds = np.log(average) - np.log1p(-average)
    return sigmoid(model["calibration_intercept"] + model["calibration_slope"] * log_odds)


def validate_patient(model: dict, patient: dict) -> np.ndarray:
    if any(patient.get(feature) is None for feature in model["features"]):
        raise ValueError("Please complete all five inputs before predicting.")
    try:
        values = np.array([patient[feature] for feature in model["features"]], dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("All numeric inputs must contain valid numbers.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite values cannot be used for prediction.")
    age, moca, depression, gfr, duration = values
    if not 0 <= age <= 120:
        raise ValueError("Age must be between 0 and 120 years.")
    if not 0 <= moca <= 30:
        raise ValueError("MOCA Score must be between 0 and 30.")
    if depression not in (0, 1):
        raise ValueError("Depression must be No (0) or Yes (1).")
    if not 0 <= gfr <= 300:
        raise ValueError("GFR must be between 0 and 300, in the training-data unit.")
    if not 0 < duration <= 1440:
        raise ValueError("Operation Time must be between 0 and 1440 minutes, excluding 0.")
    return values


def explain_exact(model: dict, values) -> dict:
    """Exact Shapley values for all 32 coalitions against one fixed reference profile."""
    values = np.asarray(values, dtype=float)
    background = np.asarray(model["reference_profile"], dtype=float)
    count = len(values)
    coalitions = np.tile(background, (1 << count, 1))
    for mask in range(1 << count):
        for feature in range(count):
            if mask & (1 << feature):
                coalitions[mask, feature] = values[feature]
    scores = predict_array(model, coalitions)
    contributions = np.zeros(count)
    for feature in range(count):
        for mask in range(1 << count):
            if mask & (1 << feature):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            contributions[feature] += weight * (scores[mask | (1 << feature)] - scores[mask])
    error = abs(float(scores[0] + contributions.sum() - scores[-1]))
    if error > 1e-10:
        raise ArithmeticError("The SHAP additivity check failed.")
    return {
        "base_value": float(scores[0]),
        "probability": float(scores[-1]),
        "shap_values": contributions.tolist(),
        "additivity_error": error,
    }


def evaluate_patient(model: dict, patient: dict, operating_point: str = "primary") -> dict:
    values = validate_patient(model, patient)
    if operating_point not in model["operating_points"]:
        raise ValueError("Unknown operating point.")
    selected = model["operating_points"][operating_point]
    explanation = explain_exact(model, values)
    clipped = np.clip(values, model["clip_lower"], model["clip_upper"])
    outside = [FEATURE_LABELS[f] for i, f in enumerate(model["features"])
               if values[i] < model["observed_lower"][i] or values[i] > model["observed_upper"][i]]
    adjusted = [FEATURE_LABELS[f] for i, f in enumerate(model["features"])
                if abs(values[i] - clipped[i]) > 1e-10]
    return {
        **explanation,
        "model": model["model"],
        "version": model["version"],
        "operating_point": operating_point,
        "operating_point_label": selected["label"],
        "threshold": selected["threshold"],
        "predicted_class": int(explanation["probability"] >= selected["threshold"]),
        "features": model["features"],
        "inputs": values.tolist(),
        "model_inputs": clipped.tolist(),
        "reference_profile": model["reference_profile"],
        "outside_training_range": outside,
        "clipped_features": adjusted,
    }
