"""Per-task evaluators: image tasks + Rello tabular/aggregate tasks."""
from __future__ import annotations

import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

from scripts.common import IMG_SUFFIXES
from scripts.evaluation.plots import (
    PLOTS_DIR,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
)


def resolve_auto_cap(image_dir):
    """Dynamic per-class cap so evaluation stays fast on huge datasets."""
    AUTO_TOTAL_BUDGET = 3000     # evaluation is for metrics — smaller budget than training
    MIN_CAP_PER_CLASS = 100
    MAX_CAP_PER_CLASS = 600

    if not os.path.isdir(image_dir):
        return 500
    counts = {}
    for cls in sorted(os.listdir(image_dir)):
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        counts[cls] = sum(
            1 for f in os.listdir(cls_dir)
            if f.lower().endswith(IMG_SUFFIXES)
        )
    n_classes = len(counts)
    total = sum(counts.values())
    if n_classes == 0 or total <= AUTO_TOTAL_BUDGET:
        return 500
    cap = max(MIN_CAP_PER_CLASS, min(MAX_CAP_PER_CLASS, AUTO_TOTAL_BUDGET // n_classes))
    print(f"  Auto-cap: {cap} images/class (evaluation budget {AUTO_TOTAL_BUDGET})")
    return cap


def _report(task, model_name, y_true, y_pred, y_proba=None, class_names=None):
    """Build evaluation dict and print summary."""
    class_names = class_names or [str(c) for c in np.unique(y_true)]
    n_classes = len(class_names)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Binary sensitivity/specificity
    sens = spec = auc = None
    if n_classes == 2 and cm.shape == (2, 2):
        from sklearn.metrics import roc_auc_score

        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        if y_proba is not None and y_proba.ndim >= 1:
            try:
                auc = float(roc_auc_score(y_true, y_proba))
            except Exception:
                auc = None

    report = {
        'task': task,
        'model': model_name,
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'sensitivity': round(sens, 4) if sens is not None else None,
        'specificity': round(spec, 4) if spec is not None else None,
        'roc_auc': round(auc, 4) if auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'class_distribution': {str(c): int((y_true == c).sum()) for c in np.unique(y_true)},
        'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    print(f"    Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}", end="")
    if sens is not None:
        print(f" | Sens: {sens:.4f} | Spec: {spec:.4f} | AUC: {auc}", end="")
    print()
    return report


# ============================================================
# IMAGE TASKS
# ============================================================

def _load_image_dataset(image_dir, max_per_class=500):
    """Load images, preprocess, extract features. Returns X, y, class_names."""
    import cv2
    from app.ml.feature_extraction import extract_all_features
    from app.ml.preprocessing import preprocess_pipeline

    if not os.path.exists(image_dir):
        return None, None, None

    classes = sorted([d for d in os.listdir(image_dir)
                      if os.path.isdir(os.path.join(image_dir, d))])
    if not classes:
        return None, None, None

    X, y = [], []
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(image_dir, class_name)
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(IMG_SUFFIXES)]
        if max_per_class:
            images = images[:max_per_class]
        for img_name in images:
            img = cv2.imread(os.path.join(class_dir, img_name))
            if img is None:
                continue
            processed = preprocess_pipeline(img)
            features, _ = extract_all_features(processed)
            X.append(features)
            y.append(label_idx)

    if len(X) == 0:
        return None, None, None
    return np.array(X), np.array(y), classes


def evaluate_image_task(task_name, data_dir, model_dir, max_per_class=500):
    """Evaluate all saved models for an image-based task."""
    from app.ml.classifiers import load_model

    print(f"\n{'='*60}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*60}")

    image_dir = os.path.join(data_dir, task_name)
    X, y, class_names = _load_image_dataset(image_dir, max_per_class=max_per_class)

    if X is None:
        print(f"  [SKIP] No data found at {image_dir}")
        return []

    print(f"  Loaded {len(X)} samples | Classes: {class_names} | Features: {X.shape[1]}")
    plot_class_distribution(y, class_names, f'{task_name} - Class Distribution',
                            PLOTS_DIR / f'{task_name}_class_distribution.png')

    if len(np.unique(y)) < 2:
        print("  [SKIP] Only one class present - cannot evaluate.")
        return []

    # Train/test split (same seed as training)
    min_class = min(np.bincount(y))
    stratify = y if min_class >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    model_files = [
        f for f in glob.glob(os.path.join(model_dir, f'{task_name}_*.pkl'))
        if not any(x in os.path.basename(f) for x in ['feature_names', 'pca'])
    ]

    if not model_files:
        print(f"  [SKIP] No model files found matching {task_name}_*.pkl")
        return []

    results = []
    for mf in sorted(model_files):
        model_name = os.path.basename(mf).replace(f'{task_name}_', '').replace('.pkl', '')
        print(f"  -> Evaluating {model_name} ...")
        try:
            model = load_model(mf)
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, 'predict_proba') and len(class_names) == 2:
                proba = model.predict_proba(X_test)
                y_proba = proba[:, 1] if proba.ndim > 1 else proba

            r = _report(task_name, model_name, y_test, y_pred, y_proba, class_names)
            results.append(r)

            plot_confusion_matrix(
                r['confusion_matrix'], class_names,
                f'{task_name} - {model_name}', PLOTS_DIR / f'cm_{task_name}_{model_name}.png'
            )
            if y_proba is not None:
                plot_roc_curve(y_test, y_proba,
                               f'{task_name} - {model_name} ROC',
                               PLOTS_DIR / f'roc_{task_name}_{model_name}.png')
        except Exception as e:
            print(f"     [ERROR] {e}")

    return results


# ============================================================
# TABULAR TASKS
# ============================================================

def _load_rellos_tabular(tabular_dir):
    """Load and combine Rello CSVs."""
    files = glob.glob(os.path.join(tabular_dir, '*.csv'))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=';', engine='python')
            if len(df.columns) == 1:
                df = pd.read_csv(f, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(f, sep=None, engine='python')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _evaluate_tabular_models(task_name, data, target_builder, feature_frame,
                             data_dir, model_dir, label_names):
    """Shared evaluation path for both Rello tabular tasks."""
    from app.ml.classifiers import load_model

    y = target_builder(data)

    min_class = min(np.bincount(y))
    stratify = y if min_class >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame, y, test_size=0.2, random_state=42, stratify=stratify
    )

    imputer_path = os.path.join(model_dir, f'{task_name}_imputer.pkl')
    scaler_path = os.path.join(model_dir, f'{task_name}_scaler.pkl')
    if os.path.exists(imputer_path):
        imp = joblib.load(imputer_path)
        X_test = imp.transform(X_test)
    if os.path.exists(scaler_path):
        scl = joblib.load(scaler_path)
        X_test = scl.transform(X_test)

    plot_class_distribution(y_test, label_names,
                            f'{task_name} - Class Distribution',
                            PLOTS_DIR / f'{task_name}_class_distribution.png')

    model_files = [
        f for f in glob.glob(os.path.join(model_dir, f'{task_name}_*.pkl'))
        if not any(x in os.path.basename(f) for x in ['feature_names', 'imputer', 'scaler'])
    ]

    results = []
    for mf in sorted(model_files):
        model_name = os.path.basename(mf).replace(f'{task_name}_', '').replace('.pkl', '')
        print(f"  -> Evaluating {model_name} ...")
        try:
            model = load_model(mf)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            r = _report(task_name, model_name, y_test, y_pred, y_proba, label_names)
            results.append(r)
            plot_confusion_matrix(r['confusion_matrix'], label_names,
                                  f'{task_name} - {model_name}',
                                  PLOTS_DIR / f'cm_{task_name}_{model_name}.png')
            if y_proba is not None:
                plot_roc_curve(y_test, y_proba,
                               f'{task_name} - {model_name}',
                               PLOTS_DIR / f'roc_{task_name}_{model_name}.png')
        except Exception as e:
            print(f"     [ERROR] {e}")
    return results


def evaluate_dyslexia_tabular(data_dir, model_dir):
    """Evaluate dyslexia_tabular models (5-feature reading metrics)."""
    print(f"\n{'='*60}")
    print("TASK: DYSLEXIA_TABULAR")
    print(f"{'='*60}")

    tabular_dir = os.path.join(data_dir, 'tabular')
    data = _load_rellos_tabular(tabular_dir)
    if data is None:
        print("  [SKIP] No CSV files in tabular/")
        return []

    target_col = None
    for col in data.columns:
        if col.lower() in ('dyslexia', 'label', 'target', 'class', 'diagnosis'):
            target_col = col
            break
    if target_col is None:
        target_col = data.columns[-1]

    y_raw = data[target_col]
    if not pd.api.types.is_numeric_dtype(y_raw):
        uniques = sorted(y_raw.dropna().unique().tolist())
        positive = next((u for u in uniques
                         if str(u).strip().lower() in ('yes', '1', 'true', 'dyslexia', 'dyslexic')),
                        uniques[-1])
        y = (y_raw.astype(str).str.strip() == str(positive)).astype(int).values
    else:
        y = y_raw.values

    X = data.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    print(f"  Loaded {len(X)} samples | Features: {X.shape[1]} "
          f"| Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    if len(np.unique(y)) < 2:
        print("  [SKIP] Only one class")
        return []

    return _evaluate_tabular_models(
        'dyslexia_tabular', data,
        target_builder=lambda d: y,
        feature_frame=X,
        data_dir=data_dir, model_dir=model_dir,
        label_names=['Normal', 'Dyslexia'],
    )


def evaluate_dyslexia_aggregate(data_dir, model_dir):
    """Evaluate dyslexia_aggregate models (8-feature aggregates)."""
    print(f"\n{'='*60}")
    print("TASK: DYSLEXIA_AGGREGATE")
    print(f"{'='*60}")

    tabular_dir = os.path.join(data_dir, 'tabular')
    data = _load_rellos_tabular(tabular_dir)
    if data is None:
        print("  [SKIP] No CSV files")
        return []

    if 'Dyslexia' not in data.columns:
        print("  [SKIP] 'Dyslexia' column missing")
        return []

    # Build aggregates exactly like training
    agg = pd.DataFrame()
    gender_raw = data['Gender'].astype(str).str.strip().str.lower()
    agg['gender'] = gender_raw.map({'m': 1, 'male': 1, '1': 1,
                                    'f': 0, 'female': 0, '0': 0}).fillna(0)
    agg['age'] = pd.to_numeric(data['Age'], errors='coerce')

    agg['total_clicks'] = data[[c for c in data.columns if c.startswith('Clicks')]].sum(axis=1, skipna=True)
    agg['total_hits'] = data[[c for c in data.columns if c.startswith('Hits')]].sum(axis=1, skipna=True)
    agg['total_misses'] = data[[c for c in data.columns if c.startswith('Misses')]].sum(axis=1, skipna=True)
    agg['total_score'] = data[[c for c in data.columns if c.startswith('Score')]].sum(axis=1, skipna=True)
    agg['mean_accuracy'] = data[[c for c in data.columns if c.startswith('Accuracy')]].mean(axis=1, skipna=True)
    agg['mean_missrate'] = data[[c for c in data.columns if c.startswith('Missrate')]].mean(axis=1, skipna=True)

    valid = agg.notna().any(axis=1) & pd.notna(data['Dyslexia'])
    agg = agg[valid]
    y_full = (data.loc[valid, 'Dyslexia'].astype(str).str.strip().str.lower() == 'yes').astype(int).values
    X = agg.fillna(agg.median())

    print(f"  Loaded {len(X)} samples | Features: {X.shape[1]} "
          f"| Classes: {dict(zip(*np.unique(y_full, return_counts=True)))}")

    if len(np.unique(y_full)) < 2:
        print("  [SKIP] Only one class")
        return []

    def target_builder(d):
        return y_full

    return _evaluate_tabular_models(
        'dyslexia_aggregate', data,
        target_builder=target_builder,
        feature_frame=X,
        data_dir=data_dir, model_dir=model_dir,
        label_names=['Normal', 'Dyslexia'],
    )
