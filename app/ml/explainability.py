"""Explainable-AI helpers.

Provides SHAP, LIME, and permutation-importance wrappers that return
JSON-serialisable data dictionaries instead of matplotlib figures.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

# Optional dependencies – import gracefully so the rest of the
# application still works when they are not installed.
try:
    import shap  # type: ignore[import-untyped]

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore[import-untyped]

    HAS_LIME = True
except ImportError:
    HAS_LIME = False


def get_shap_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    model_type: str = "tree",
) -> Optional[np.ndarray]:
    """Compute SHAP values for *X*.

    Args:
        model: Fitted estimator.
        X: Feature matrix ``(n_samples, n_features)``.
        feature_names: Human-readable feature labels.
        model_type: ``'tree'`` for tree-based explainer,
            ``'kernel'`` for the model-agnostic KernelExplainer.

    Returns:
        SHAP values array or *None* when ``shap`` is unavailable.
    """
    if not HAS_SHAP:
        warnings.warn("shap is not installed; returning None", stacklevel=2)
        return None

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X, min(100, X.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X)
    return shap_values


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract feature importances from a model.

    Prefers native ``feature_importances_`` (tree models) and falls
    back to permutation importance when ``X`` and ``y`` are provided.

    Args:
        model: Fitted estimator.
        feature_names: Feature labels.
        X: Validation features (needed for permutation fallback).
        y: Validation labels (needed for permutation fallback).

    Returns:
        ``{feature_name: importance_score}`` dict.
    """
    importances: np.ndarray

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).flatten()
    elif X is not None and y is not None:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = result.importances_mean
    else:
        importances = np.zeros(len(feature_names))

    names = [str(n) for n in feature_names[: len(importances)]]
    return dict(zip(names, importances.tolist()))


def get_lime_explanation(
    model: Any,
    X_train: np.ndarray,
    instance: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a LIME explanation for a single prediction.

    Args:
        model: Fitted estimator with ``predict_proba``.
        X_train: Training data used to fit the LIME explainer.
        instance: Single sample ``(1, n_features)``.
        feature_names: Feature labels.
        class_names: Optional class labels.

    Returns:
        Dictionary with ``prediction``, ``local_exp`` (list of
        ``(class_idx, [(feat_idx, weight), ...])``), and
        ``intercept`` – or *None* when ``lime`` is unavailable.
    """
    if not HAS_LIME:
        warnings.warn("lime is not installed; returning None", stacklevel=2)
        return None

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
    )

    exp = explainer.explain_instance(
        instance.flatten(), model.predict_proba, num_features=len(feature_names)
    )

    return {
        "prediction": int(exp.predict_proba.argmax()) if hasattr(exp, "predict_proba") else None,
        "local_exp": exp.local_exp,
        "intercept": exp.intercept,
    }


def generate_shap_plot_data(
    shap_values: Any,
    feature_names: List[str],
    instance_idx: int = 0,
) -> Dict[str, Any]:
    """Convert raw SHAP values into a JSON-serialisable summary dict.

    Args:
        shap_values: Output of :func:`get_shap_explanation`.
        feature_names: Feature labels.
        instance_idx: Which sample to extract when values are 3-D
            (binary models may return a list of two arrays).

    Returns:
        ``{"features": [...], "values": [...]}`` sorted by absolute
        magnitude.
    """
    if shap_values is None:
        return {"features": [], "values": []}

    sv = np.asarray(shap_values)

    if sv.ndim == 3:
        vals = sv[instance_idx, :, 1]
    elif sv.ndim == 2:
        vals = sv[instance_idx, :]
    else:
        vals = sv.flatten()

    names = [str(n) for n in feature_names[: len(vals)]]
    order = np.argsort(np.abs(vals))[::-1]

    return {
        "features": [names[i] for i in order],
        "values": [float(vals[i]) for i in order],
    }


def generate_feature_importance_plot_data(
    importances: Dict[str, float],
) -> Dict[str, Any]:
    """Sort importances descending and return a plot-ready dict.

    Args:
        importances: ``{feature_name: score}`` mapping.

    Returns:
        ``{"features": [...], "values": [...]}`` sorted by importance
        (descending).
    """
    sorted_items = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "features": [k for k, _ in sorted_items],
        "values": [float(v) for _, v in sorted_items],
    }
