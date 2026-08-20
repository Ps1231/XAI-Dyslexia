"""Classifier training, evaluation, and persistence.

Wraps scikit-learn estimators behind a unified interface for easy
experimentation and model selection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def get_classifiers() -> Dict[str, Any]:
    """Return a mapping of classifier names to fresh estimator instances.

    Includes:
        - ``svm_linear`` – Linear SVC
        - ``svm_rbf`` – RBF-kernel SVC
        - ``random_forest`` – Random Forest (200 trees)
        - ``gradient_boosting`` – Gradient Boosting (100 trees)
        - ``mlp`` – Multi-layer Perceptron
    """
    return {
        "svm_linear": SVC(kernel="linear", probability=True),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
        ),
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_name: str = "random_forest",
) -> Any:
    """Train a single classifier and return the fitted model.

    Args:
        X_train: Feature matrix of shape ``(n_samples, n_features)``.
        y_train: Label vector of shape ``(n_samples,)``.
        classifier_name: Key into :func:`get_classifiers`.

    Returns:
        Fitted scikit-learn estimator.

    Raises:
        ValueError: If *classifier_name* is not recognised.
    """
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
    """Evaluate a fitted model on a test set.

    Args:
        model: Fitted scikit-learn estimator.
        X_test: Feature matrix.
        y_test: Ground-truth labels.

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``confusion_matrix``, and ``classification_report``.
    """
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
) -> Dict[str, Any]:
    """Train every classifier and collect evaluation metrics.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        ``{classifier_name: {"model": fitted_model, "metrics": {...}}}``.
    """
    results: Dict[str, Any] = {}
    for name in get_classifiers():
        try:
            model = train_model(X_train, y_train, classifier_name=name)
            metrics = evaluate_model(model, X_test, y_test)
            results[name] = {"model": model, "metrics": metrics}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"model": None, "metrics": {}, "error": str(exc)}
    return results


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a model to disk with joblib.

    Args:
        model: Fitted estimator.
        path: Destination file path.

    Returns:
        Resolved :class:`~pathlib.Path` of the saved file.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest.resolve()


def load_model(path: str | Path) -> Any:
    """Load a previously saved model.

    Args:
        path: Path to the joblib file.

    Returns:
        Deserialised estimator.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {src}")
    return joblib.load(src)
