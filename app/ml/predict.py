"""High-level prediction pipeline.

Takes a raw image (or behavioural feature dictionary for dyslexia),
runs preprocessing, feature extraction, model inference, and
explainability in one call.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.ml.classifiers import load_model
from app.ml.explainability import (
    generate_feature_importance_plot_data,
    generate_shap_plot_data,
    get_feature_importance,
    get_lime_explanation,
    get_shap_explanation,
)
from app.ml.feature_extraction import extract_all_features
from app.ml.preprocessing import load_image, preprocess_pipeline


def predict_dysgraphia(
    image_path: str,
    model_path: str,
    X_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """End-to-end dysgraphia prediction from a handwriting image.

    Steps:
        1. Load and preprocess the image.
        2. Extract feature vector.
        3. Run model prediction with confidence.
        4. Compute SHAP, LIME, and feature-importance explanations.

    Args:
        image_path: Path to the handwriting image.
        model_path: Path to a serialised scikit-learn model.
        X_train: Optional training data needed for LIME / permutation
            importance.  When *None* those explanations are skipped.

    Returns:
        Dictionary with keys ``prediction``, ``confidence``,
        ``features``, ``feature_names``, ``shap_values``,
        ``feature_importance``, ``lime_explanation``.
    """
    result: Dict[str, Any] = {
        "prediction": None,
        "confidence": None,
        "features": None,
        "feature_names": None,
        "shap_values": None,
        "feature_importance": None,
        "lime_explanation": None,
    }

    # 1 – Load & preprocess
    raw = load_image(image_path)
    if raw is None:
        result["error"] = f"Could not load image: {image_path}"
        return result

    processed = preprocess_pipeline(raw)

    # 2 – Extract features
    feature_vector, feature_names = extract_all_features(processed)
    result["features"] = feature_vector.tolist()
    result["feature_names"] = feature_names

    # 3 – Predict
    model = load_model(model_path)
    X = feature_vector.reshape(1, -1)
    prediction = int(model.predict(X)[0])
    result["prediction"] = prediction

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["confidence"] = float(proba.max())
        result["probabilities"] = proba.tolist()
    else:
        result["confidence"] = 1.0

    # 4 – Explainability
    # SHAP
    try:
        model_type = "tree" if hasattr(model, "estimators_") else "kernel"
        shap_vals = get_shap_explanation(model, X, feature_names, model_type=model_type)
        result["shap_values"] = generate_shap_plot_data(
            shap_vals, feature_names, instance_idx=0
        )
    except Exception:  # noqa: BLE001
        result["shap_values"] = None

    # Feature importance
    try:
        imp = get_feature_importance(
            model,
            feature_names,
            X=X_train if X_train is not None else None,
            y=None,
        )
        result["feature_importance"] = generate_feature_importance_plot_data(imp)
    except Exception:  # noqa: BLE001
        result["feature_importance"] = None

    # LIME
    if X_train is not None:
        try:
            result["lime_explanation"] = get_lime_explanation(
                model,
                X_train,
                X,
                feature_names,
                class_names=["Normal", "Dysgraphia"],
            )
        except Exception:  # noqa: BLE001
            result["lime_explanation"] = None

    return result


def predict_dyslexia(
    features_dict: Dict[str, float],
    model_path: str,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Predict dyslexia from behavioural / performance features.

    Unlike :func:`predict_dysgraphia`, this function does **not**
    perform image preprocessing – it expects a pre-computed feature
    dictionary (e.g. reading speed, error rate, phoneme accuracy …).

    Args:
        features_dict: ``{feature_name: value}``.
        model_path: Path to a serialised model.
        X_train: Optional training data for LIME explanations.
        feature_names: Ordered list of feature names.  When *None* the
            keys of *features_dict* are used (order depends on dict
            ordering guarantee of the runtime).

    Returns:
        Same structure as :func:`predict_dysgraphia`.
    """
    result: Dict[str, Any] = {
        "prediction": None,
        "confidence": None,
        "features": None,
        "feature_names": None,
        "shap_values": None,
        "feature_importance": None,
        "lime_explanation": None,
    }

    if feature_names is None:
        feature_names = list(features_dict.keys())

    vector = np.array([features_dict.get(n, 0.0) for n in feature_names], dtype=np.float64)
    result["features"] = vector.tolist()
    result["feature_names"] = feature_names

    model = load_model(model_path)
    X = vector.reshape(1, -1)

    prediction = int(model.predict(X)[0])
    result["prediction"] = prediction

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["confidence"] = float(proba.max())
        result["probabilities"] = proba.tolist()
    else:
        result["confidence"] = 1.0

    # Explainability
    try:
        model_type = "tree" if hasattr(model, "estimators_") else "kernel"
        shap_vals = get_shap_explanation(model, X, feature_names, model_type=model_type)
        result["shap_values"] = generate_shap_plot_data(
            shap_vals, feature_names, instance_idx=0
        )
    except Exception:  # noqa: BLE001
        result["shap_values"] = None

    try:
        imp = get_feature_importance(
            model,
            feature_names,
            X=X_train if X_train is not None else None,
            y=None,
        )
        result["feature_importance"] = generate_feature_importance_plot_data(imp)
    except Exception:  # noqa: BLE001
        result["feature_importance"] = None

    if X_train is not None:
        try:
            result["lime_explanation"] = get_lime_explanation(
                model,
                X_train,
                X,
                feature_names,
                class_names=["Normal", "Dyslexia"],
            )
        except Exception:  # noqa: BLE001
            result["lime_explanation"] = None

    return result


def get_prediction_report(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a raw prediction result into a human-readable report.

    Args:
        prediction_result: Output of :func:`predict_dysgraphia` or
            :func:`predict_dyslexia`.

    Returns:
        Dictionary with ``summary``, ``confidence_pct``,
        ``top_shap_features``, and ``top_importance_features``.
    """
    pred = prediction_result.get("prediction")
    conf = prediction_result.get("confidence", 0.0)

    label_map = {
        "dysgraphia": {0: "Normal", 1: "Dysgraphia"},
        "dyslexia": {0: "Normal", 1: "Dyslexia"},
    }

    # Auto-detect which task based on probabilities key presence isn't
    # reliable, so default label lookup uses both.
    task_labels = label_map.get("dysgraphia", {})
    readable = task_labels.get(pred, f"Class {pred}") if pred is not None else "Unknown"

    report: Dict[str, Any] = {
        "summary": f"Prediction: {readable}",
        "confidence_pct": round((conf or 0.0) * 100, 2),
        "raw_prediction": pred,
    }

    # Top SHAP features
    shap_data = prediction_result.get("shap_values")
    if isinstance(shap_data, dict):
        feats = shap_data.get("features", [])[:5]
        vals = shap_data.get("values", [])[:5]
        report["top_shap_features"] = [
            {"feature": f, "shap_value": v} for f, v in zip(feats, vals)
        ]
    else:
        report["top_shap_features"] = []

    # Top importance features
    imp_data = prediction_result.get("feature_importance")
    if isinstance(imp_data, dict):
        feats = imp_data.get("features", [])[:5]
        vals = imp_data.get("values", [])[:5]
        report["top_importance_features"] = [
            {"feature": f, "importance": v} for f, v in zip(feats, vals)
        ]
    else:
        report["top_importance_features"] = []

    return report
