"""Model trainers: dynamic image capping, image + tabular task training,
and fingerprint-based up-to-date guards."""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import joblib
import numpy as np

from scripts.common import console, dir_fingerprint, load_state, save_state, IMG_SUFFIXES

# Dynamic sampling budget — datasets larger than this are down-sampled per class.
AUTO_TOTAL_BUDGET = 6000
MIN_CAP_PER_CLASS = 250      # never sample below this (keep minority classes viable)
MAX_CAP_PER_CLASS = 3000     # never waste time above this per class


# -------------------------------------------------------------
# Idempotency helpers
# -------------------------------------------------------------

def _model_artifacts_exist(output_dir, name) -> bool:
    """True when at least one trained estimator + feature names are saved."""
    out = Path(output_dir)
    if not (out / f"{name}_feature_names.pkl").exists():
        return False
    aux = ("feature_names", "imputer", "scaler", "pca")
    return any(
        not any(x in p.name for x in aux)
        for p in out.glob(f"{name}_*.pkl")
    )


def _task_up_to_date(src_dir, output_dir, key, force) -> bool:
    """True when source data is unchanged since the last successful training."""
    if force:
        return False
    fp = dir_fingerprint(src_dir)
    state = load_state(Path(output_dir) / ".state", f"train_{key}")
    if not (state and fp and state.get("fingerprint") == fp):
        return False
    if not _model_artifacts_exist(output_dir, key):
        return False
    console.print(f"[yellow][skip] {key}: data unchanged since last training")
    return True


def _mark_task_done(src_dir, output_dir, key):
    fp = dir_fingerprint(src_dir)
    if fp:
        save_state(Path(output_dir) / ".state", f"train_{key}", {"fingerprint": fp})


# -------------------------------------------------------------
# Dynamic auto-cap
# -------------------------------------------------------------

def resolve_auto_cap(image_dir):
    """Dynamically pick a per-class image cap so the total stays within budget.

    Returns None when the dataset is small enough to use in full.
    """
    if not os.path.isdir(image_dir):
        return None

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
        return None

    cap = AUTO_TOTAL_BUDGET // n_classes
    cap = max(MIN_CAP_PER_CLASS, min(MAX_CAP_PER_CLASS, cap))
    print(f"  Auto-cap: {cap} images/class "
          f"({n_classes} classes x {total} total -> ~{min(total, cap * n_classes)} used; "
          f"budget {AUTO_TOTAL_BUDGET})")
    return cap


# -------------------------------------------------------------
# Image-task trainer
# -------------------------------------------------------------

def train_image_dataset(name, image_dir, output_dir,
                        max_images_per_class="auto", use_pca=False):
    """Reusable training block for any handwriting-image dataset.

    max_images_per_class: int for a fixed cap, "auto" for dynamic budget-based
    capping, or None to always use every image.
    """
    import cv2
    from app.ml.classifiers import train_and_evaluate_all, save_model
    from app.ml.feature_extraction import extract_pca_features
    from scripts.training.loaders import load_image_dataset

    print(f"\n=== Training {name} Models ===")
    if not os.path.exists(image_dir):
        print(f"  {name} data directory not found: {image_dir}")
        return

    if max_images_per_class == "auto":
        max_images_per_class = resolve_auto_cap(image_dir)

    t0 = time.time()
    X, y, feature_names = load_image_dataset(image_dir, max_images_per_class=max_images_per_class)
    if X is None or len(X) == 0:
        print(f"  No valid images found for {name} training")
        return

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"  Feature matrix: {X.shape}  (extracted in {time.time()-t0:.1f}s)")
    print(f"  Classes in data: {n_classes}")

    if n_classes < 2:
        print(f"  ERROR: Cannot train with only {n_classes} class(es).")
        print(f"  Check that {image_dir} has at least 2 class subfolders with images.")
        return

    # Skip slow models for large high-dimensional datasets
    skip_models = []
    if n_samples > 1000 or n_features > 5000:
        skip_models.extend(["svm_linear", "svm_rbf"])
        print(f"  NOTE: Large dataset ({n_samples} samples, {n_features} features).")
        print(f"        Skipping both SVMs (too slow). Training RF, GB, MLP, LR, DT only.")

    from sklearn.model_selection import train_test_split

    # Guard stratified split for tiny classes
    min_class_count = min(np.bincount(y))
    if min_class_count < 2:
        print(f"  WARNING: Class with only {min_class_count} sample(s). Using simple split (no stratify).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"  Train/test split: {len(X_train)} train, {len(X_test)} test")
    print(f"  Starting model training (this may take 2-5 minutes per model)...")

    pca_model = None
    if use_pca and n_features > 50:
        print("  Applying PCA...")
        X_train, X_test, pca_model = extract_pca_features(X_train, X_test, n_components=0.95)

    results = train_and_evaluate_all(X_train, X_test, y_train, y_test, skip_models=skip_models)

    for model_name, result in results.items():
        if result.get('model') is None:
            print(f"\n{model_name}: FAILED - {result.get('error', 'unknown error')}")
            continue

        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        model = result['model']
        save_model(model, os.path.join(
            output_dir, f'{name}_{model_name.lower().replace(" ", "_")}.pkl'))

    if pca_model is not None:
        joblib.dump(pca_model, os.path.join(output_dir, f'{name}_pca.pkl'))

    joblib.dump(feature_names, os.path.join(output_dir, f'{name}_feature_names.pkl'))
    print(f"\n{name} models saved to {output_dir}")


# -------------------------------------------------------------
# Tabular aggregate trainer (Rello visual search)
# -------------------------------------------------------------

def train_dyslexia_aggregate(tabular_dir, output_dir):
    """Train aggregate-feature model from Rello visual search data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from app.ml.classifiers import train_and_evaluate_all, save_model

    print("\n=== Training Dyslexia Aggregate (Rello) Models ===")

    files = glob.glob(os.path.join(tabular_dir, '*.csv'))
    if not files:
        print(f"  No CSV files found in {tabular_dir}")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=';', engine='python')
            if len(df.columns) == 1:
                df = pd.read_csv(f, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(f, sep=None, engine='python')
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(data)} rows from {len(files)} file(s)")

    if 'Dyslexia' not in data.columns:
        print("  ERROR: 'Dyslexia' column not found in tabular data")
        return

    y = data['Dyslexia']

    # --- AGGREGATE FEATURES ---
    agg = pd.DataFrame()

    gender_raw = data['Gender'].astype(str).str.strip().str.lower()
    agg['gender'] = gender_raw.map({'m': 1, 'male': 1, '1': 1,
                                    'f': 0, 'female': 0, '0': 0}).fillna(0)

    print(f"  Gender mapping check: {dict(zip(['F/0', 'M/1'], [int((agg['gender']==0).sum()), int((agg['gender']==1).sum())]))}")

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

    print(f"  Aggregate features: {list(agg.columns)}")
    print(f"  Shape: {agg.shape}")

    valid_mask = agg.notna().any(axis=1) & y.notna()
    agg = agg[valid_mask]
    y = y[valid_mask]

    X = agg.fillna(agg.median())
    y = (y.astype(str).str.strip().str.lower() == 'yes').astype(int).values

    print(f"  Final: X={X.shape}, y distribution={np.bincount(y)}")

    if len(np.unique(y)) < 2:
        print("  ERROR: Only one class present after filtering. Cannot train.")
        return

    min_class = min(np.bincount(y))
    if min_class < 2:
        print(f"  WARNING: Minimum class count is {min_class}. Using simple split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    joblib.dump(imputer, os.path.join(output_dir, 'dyslexia_aggregate_imputer.pkl'))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    joblib.dump(scaler, os.path.join(output_dir, 'dyslexia_aggregate_scaler.pkl'))

    results = train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test)

    for name, result in results.items():
        if result.get('model') is None:
            print(f"\n{name}: FAILED - {result.get('error', 'unknown error')}")
            continue

        metrics = result['metrics']
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        save_model(result['model'], os.path.join(output_dir, f'dyslexia_aggregate_{name.lower().replace(" ", "_")}.pkl'))

    joblib.dump(list(X.columns), os.path.join(output_dir, 'dyslexia_aggregate_feature_names.pkl'))
    print(f"\nDyslexia aggregate models saved to {output_dir}")
