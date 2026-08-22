"""Classifier training, evaluation, and persistence.

Wraps scikit-learn estimators behind a unified interface for easy
experimentation and model selection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC


def _n_jobs() -> int:
    """Limit parallel jobs to avoid system thrashing on large datasets."""
    return min(4, os.cpu_count() or 1)


def get_classifiers() -> Dict[str, Any]:
    """Return a mapping of classifier names to fresh estimator instances.

    v4 CHANGES:
    - Added LogisticRegression and DecisionTreeClassifier for inherent interpretability.
    - n_jobs limited to 4 max to prevent process thrashing.
    - RF uses 100 trees by default (faster on high-dim data).
    - MLP uses smaller hidden layers for faster convergence.
    """
    return {
        "svm_linear": CalibratedClassifierCV(
            LinearSVC(random_state=42, max_iter=5000, dual="auto"),
            method="sigmoid",
            cv=3,
        ),
        "svm_rbf": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=42),
            method="sigmoid",
            cv=3,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=_n_jobs()
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            early_stopping=True,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=_n_jobs()
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42, max_depth=10, min_samples_leaf=5
        ),
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_name: str = "random_forest",
) -> Any:
    """Train a single classifier and return the fitted model."""
    classifiers = get_classifiers()
    if classifier_name not in classifiers:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Choose from: {list(classifiers.keys())}"
        )
    model = classifiers[classifier_name]
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a fitted model on a test set."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }


def train_and_evaluate_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    skip_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train every classifier and collect evaluation metrics.

    Args:
        skip_models: List of model names to skip.
    """
    results: Dict[str, Any] = {}
    skip_models = skip_models or []
    for name in get_classifiers():
        if name in skip_models:
            results[name] = {"model": None, "metrics": {}, "error": "skipped (dataset too large)"}
            continue
        print(f"    → Training {name} ...", flush=True)
        try:
            model = train_model(X_train, y_train, classifier_name=name)
            print(f"    → Evaluating {name} ...", flush=True)
            metrics = evaluate_model(model, X_test, y_test)
            results[name] = {"model": model, "metrics": metrics}
            print(f"    ✓ {name} done", flush=True)
        except Exception as exc:
            print(f"    ✗ {name} failed: {exc}", flush=True)
            results[name] = {"model": None, "metrics": {}, "error": str(exc)}
    return results


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a model to disk with joblib."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest.resolve()


def load_model(path: str | Path) -> Any:
    """Load a previously saved model."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {src}")
    return joblib.load(src)