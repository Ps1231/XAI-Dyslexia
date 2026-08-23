"""
Comprehensive model evaluation for ALL trained tasks.

Covers:
  • dysgraphia          (image-based, binary/multi-class)
  • dyslexia_handwriting (image-based)
  • dyslexia_synthetic   (image-based)
  • dyslexia_tabular     (Rello 5-feature tabular)
  • dyslexia_aggregate   (Rello 8-feature aggregate)

Metrics:
  Accuracy, Precision, Recall, F1, Sensitivity, Specificity, ROC-AUC
  + Confusion matrix plots, ROC curves, class distribution, per-class reports

Usage:
    python scripts/evaluate.py --task all
    python scripts/evaluate.py --task dysgraphia
    python scripts/evaluate.py --task dyslexia_tabular
    python scripts/evaluate.py --task dyslexia_aggregate --plot
"""
import os
import sys
import argparse
import glob
import json
import warnings
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.classifiers import load_model
from app.ml.feature_extraction import extract_all_features
from app.ml.preprocessing import preprocess_pipeline
import cv2

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "app" / "models"
PLOTS_DIR = PROJECT_ROOT / "evaluation_plots"
PLOTS_DIR.mkdir(exist_ok=True)

REPORTS = []


def _plot_confusion_matrix(cm, class_names, title, save_path):
    """Save a confusion matrix heatmap."""
    cm = np.array(cm)  # handle list input from JSON serialization
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(y_true, y_proba, title, save_path):
    """Save ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return auc


def _plot_class_distribution(y, class_names, title, save_path):
    """Bar chart of class distribution."""
    counts = np.bincount(y, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, counts, color=['#0891b2', '#ef4444', '#f59e0b', '#10b981'][:len(class_names)])
    ax.set_title(title)
    ax.set_ylabel('Count')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                str(c), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
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
    print(f"\n{'='*60}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*60}")

    image_dir = os.path.join(data_dir, task_name)
    X, y, class_names = _load_image_dataset(image_dir, max_per_class=max_per_class)

    if X is None:
        print(f"  [SKIP] No data found at {image_dir}")
        return []

    print(f"  Loaded {len(X)} samples | Classes: {class_names} | Features: {X.shape[1]}")
    _plot_class_distribution(y, class_names, f'{task_name} — Class Distribution',
                             PLOTS_DIR / f'{task_name}_class_distribution.png')

    if len(np.unique(y)) < 2:
        print("  [SKIP] Only one class present — cannot evaluate.")
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

            _plot_confusion_matrix(
                r['confusion_matrix'], class_names,
                f'{task_name} — {model_name}', PLOTS_DIR / f'cm_{task_name}_{model_name}.png'
            )
            if y_proba is not None:
                _plot_roc_curve(y_test, y_proba,
                                f'{task_name} — {model_name} ROC',
                                PLOTS_DIR / f'roc_{task_name}_{model_name}.png')
        except Exception as e:
            print(f"     [ERROR] {e}")

    return results


# ============================================================
# TABULAR TASKS
# ============================================================

def _load_rellos_tabular(tabular_dir):
    """Load and combine Rello CSVs."""
    import pandas as pd
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
        positive = None
        for u in uniques:
            if str(u).strip().lower() in ('yes', '1', 'true', 'dyslexia', 'dyslexic'):
                positive = u
                break
        if positive is None:
            positive = uniques[-1]
        y = (y_raw.astype(str).str.strip() == str(positive)).astype(int).values
    else:
        y = y_raw.values

    X = data.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    print(f"  Loaded {len(X)} samples | Features: {X.shape[1]} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    if len(np.unique(y)) < 2:
        print("  [SKIP] Only one class")
        return []

    min_class = min(np.bincount(y))
    stratify = y if min_class >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Load artifacts
    imputer_path = os.path.join(model_dir, 'dyslexia_tabular_imputer.pkl')
    scaler_path = os.path.join(model_dir, 'dyslexia_tabular_scaler.pkl')
    if os.path.exists(imputer_path):
        imputer = joblib.load(imputer_path)
        X_test = imputer.transform(X_test)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)

    _plot_class_distribution(y_test, ['Normal', 'Dyslexia'],
                             'Dyslexia Tabular — Class Distribution',
                             PLOTS_DIR / 'dyslexia_tabular_class_distribution.png')

    model_files = [
        f for f in glob.glob(os.path.join(model_dir, 'dyslexia_tabular_*.pkl'))
        if not any(x in os.path.basename(f) for x in ['feature_names', 'imputer', 'scaler'])
    ]

    results = []
    for mf in sorted(model_files):
        model_name = os.path.basename(mf).replace('dyslexia_tabular_', '').replace('.pkl', '')
        print(f"  -> Evaluating {model_name} ...")
        try:
            model = load_model(mf)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            r = _report('dyslexia_tabular', model_name, y_test, y_pred, y_proba, ['Normal', 'Dyslexia'])
            results.append(r)
            _plot_confusion_matrix(r['confusion_matrix'], ['Normal', 'Dyslexia'],
                                   f'Dyslexia Tabular — {model_name}',
                                   PLOTS_DIR / f'cm_dyslexia_tabular_{model_name}.png')
            if y_proba is not None:
                _plot_roc_curve(y_test, y_proba,
                                f'Dyslexia Tabular — {model_name}',
                                PLOTS_DIR / f'roc_dyslexia_tabular_{model_name}.png')
        except Exception as e:
            print(f"     [ERROR] {e}")
    return results


def evaluate_dyslexia_aggregate(data_dir, model_dir):
    """Evaluate dyslexia_aggregate models (8-feature aggregates)."""
    print(f"\n{'='*60}")
    print("TASK: DYSLEXIA_AGGREGATE")
    print(f"{'='*60}")

    import pandas as pd
    tabular_dir = os.path.join(data_dir, 'tabular')
    data = _load_rellos_tabular(tabular_dir)
    if data is None:
        print("  [SKIP] No CSV files")
        return []

    if 'Dyslexia' not in data.columns:
        print("  [SKIP] 'Dyslexia' column missing")
        return []

    y = (data['Dyslexia'].astype(str).str.strip().str.lower() == 'yes').astype(int).values

    # Build aggregates exactly like training
    agg = pd.DataFrame()
    gender_raw = data['Gender'].astype(str).str.strip().str.lower()
    agg['gender'] = gender_raw.map({'m': 1, 'male': 1, '1': 1, 'f': 0, 'female': 0, '0': 0}).fillna(0)
    agg['age'] = pd.to_numeric(data['Age'], errors='coerce')

    click_cols = [c for c in data.columns if c.startswith('Clicks')]
    hit_cols = [c for c in data.columns if c.startswith('Hits')]
    miss_cols = [c for c in data.columns if c.startswith('Misses')]
    score_cols = [c for c in data.columns if c.startswith('Score')]
    acc_cols = [c for c in data.columns if c.startswith('Accuracy')]
    missrate_cols = [c for c in data.columns if c.startswith('Missrate')]

    agg['total_clicks'] = data[click_cols].sum(axis=1, skipna=True)
    agg['total_hits'] = data[hit_cols].sum(axis=1, skipna=True)
    agg['total_misses'] = data[miss_cols].sum(axis=1, skipna=True)
    agg['total_score'] = data[score_cols].sum(axis=1, skipna=True)
    agg['mean_accuracy'] = data[acc_cols].mean(axis=1, skipna=True)
    agg['mean_missrate'] = data[missrate_cols].mean(axis=1, skipna=True)

    valid = agg.notna().any(axis=1) & pd.notna(data['Dyslexia'])
    agg = agg[valid]
    y = y[valid]
    X = agg.fillna(agg.median())

    print(f"  Loaded {len(X)} samples | Features: {X.shape[1]} | Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    if len(np.unique(y)) < 2:
        print("  [SKIP] Only one class")
        return []

    min_class = min(np.bincount(y))
    stratify = y if min_class >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    imputer_path = os.path.join(model_dir, 'dyslexia_aggregate_imputer.pkl')
    scaler_path = os.path.join(model_dir, 'dyslexia_aggregate_scaler.pkl')
    if os.path.exists(imputer_path):
        imp = joblib.load(imputer_path)
        X_test = imp.transform(X_test)
    if os.path.exists(scaler_path):
        scl = joblib.load(scaler_path)
        X_test = scl.transform(X_test)

    _plot_class_distribution(y_test, ['Normal', 'Dyslexia'],
                             'Dyslexia Aggregate — Class Distribution',
                             PLOTS_DIR / 'dyslexia_aggregate_class_distribution.png')

    model_files = [
        f for f in glob.glob(os.path.join(model_dir, 'dyslexia_aggregate_*.pkl'))
        if not any(x in os.path.basename(f) for x in ['feature_names', 'imputer', 'scaler'])
    ]

    results = []
    for mf in sorted(model_files):
        model_name = os.path.basename(mf).replace('dyslexia_aggregate_', '').replace('.pkl', '')
        print(f"  -> Evaluating {model_name} ...")
        try:
            model = load_model(mf)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            r = _report('dyslexia_aggregate', model_name, y_test, y_pred, y_proba, ['Normal', 'Dyslexia'])
            results.append(r)
            _plot_confusion_matrix(r['confusion_matrix'], ['Normal', 'Dyslexia'],
                                   f'Dyslexia Aggregate — {model_name}',
                                   PLOTS_DIR / f'cm_dyslexia_aggregate_{model_name}.png')
            if y_proba is not None:
                _plot_roc_curve(y_test, y_proba,
                                f'Dyslexia Aggregate — {model_name}',
                                PLOTS_DIR / f'roc_dyslexia_aggregate_{model_name}.png')
        except Exception as e:
            print(f"     [ERROR] {e}")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained XAI-Dyslexia models')
    parser.add_argument('--task', choices=['dysgraphia', 'dyslexia_handwriting', 'dyslexia_synthetic',
                                            'dyslexia_tabular', 'dyslexia_aggregate', 'all'],
                        default='all')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--max-images', type=int, default=500,
                        help='Max images per class for image tasks (speed vs accuracy tradeoff)')
    args = parser.parse_args()

    all_results = []

    if args.task in ('dysgraphia', 'all'):
        all_results.extend(evaluate_image_task('dysgraphia', args.data_dir, args.model_dir, args.max_images))

    if args.task in ('dyslexia_handwriting', 'all'):
        all_results.extend(evaluate_image_task('dyslexia_handwriting', args.data_dir, args.model_dir, args.max_images))

    if args.task in ('dyslexia_synthetic', 'all'):
        all_results.extend(evaluate_image_task('dyslexia_synthetic', args.data_dir, args.model_dir, args.max_images))

    if args.task in ('dyslexia_tabular', 'all'):
        all_results.extend(evaluate_dyslexia_tabular(args.data_dir, args.model_dir))

    if args.task in ('dyslexia_aggregate', 'all'):
        all_results.extend(evaluate_dyslexia_aggregate(args.data_dir, args.model_dir))

    # Save master report
    report_path = PLOTS_DIR / 'master_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total models evaluated: {len(all_results)}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Master report: {report_path}")

    # Print summary table
    if all_results:
        print(f"\n{'Model':<40} {'Task':<20} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7} {'AUC':>7}")
        print("-" * 100)
        for r in all_results:
            auc_str = f"{r['roc_auc']:.3f}" if r['roc_auc'] is not None else "N/A"
            sens_str = f"{r['sensitivity']:.3f}" if r['sensitivity'] is not None else "N/A"
            spec_str = f"{r['specificity']:.3f}" if r['specificity'] is not None else "N/A"
            print(f"{r['model']:<40} {r['task']:<20} {r['accuracy']:>7.3f} {r['f1']:>7.3f} {sens_str:>7} {spec_str:>7} {auc_str:>7}")


if __name__ == '__main__':
    main()