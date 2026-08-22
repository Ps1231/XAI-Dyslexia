"""
Train and save ML models for dysgraphia and dyslexia detection.

Usage:
    python scripts/train_models.py --task all
    python scripts/train_models.py --task dysgraphia
    python scripts/train_models.py --task dyslexia_tabular
    python scripts/train_models.py --task dyslexia_handwriting
    python scripts/train_models.py --task dyslexia_aggregate
    python scripts/train_models.py --task dyslexia_handwriting --full
"""
import os
import sys
import argparse
import time
import random
import glob
import numpy as np
import joblib
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "app" / "models"

sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.classifiers import get_classifiers, train_and_evaluate_all, save_model
from app.ml.feature_extraction import extract_all_features, extract_pca_features
from app.ml.preprocessing import preprocess_pipeline
import cv2

DEFAULT_MAX_PER_CLASS = 2000


def load_image_dataset(data_dir, max_images_per_class=None):
    """Load images from directory structure: data_dir/class_name/image.jpg"""
    X_features = []
    y_labels = []
    feature_names = []

    classes = sorted([d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))])

    if not classes:
        print("No class directories found in", data_dir)
        return None, None, None

    print(f"Found classes: {classes}")

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        if max_images_per_class and len(images) > max_images_per_class:
            print(f"  {class_name}: {len(images)} images -> sampling {max_images_per_class}")
            random.seed(42)
            images = random.sample(images, max_images_per_class)
        else:
            print(f"  {class_name}: {len(images)} images")

        for img_name in tqdm(images, desc=f"  extracting [{class_name}]", unit="img"):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            processed = preprocess_pipeline(img)
            features, names = extract_all_features(processed)

            if features is not None and len(features) > 0:
                X_features.append(features)
                y_labels.append(label_idx)
                if not feature_names:
                    feature_names = names

    return np.array(X_features), np.array(y_labels), feature_names


def train_image_dataset(name, image_dir, output_dir, max_images_per_class=None, use_pca=False):
    """Reusable training block for any handwriting-image dataset."""
    print(f"\n=== Training {name} Models ===")
    if not os.path.exists(image_dir):
        print(f"  {name} data directory not found: {image_dir}")
        return

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
        print(f"  Skipping {name} training.")
        return

    # Skip slow models for large high-dimensional datasets
    skip_models = []
    if n_samples > 1000 or n_features > 5000:
        skip_models.extend(["svm_linear", "svm_rbf"])
        print(f"  NOTE: Large dataset ({n_samples} samples, {n_features} features).")
        print(f"        Skipping both SVMs (too slow). Training RF, GB, MLP, LR, DT only.")
        print(f"        Use --full to force SVM training (may take 10+ min).")

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
    print(f"  Starting model training (this may take 2–5 minutes per model)...")

    # Optional PCA
    pca_model = None
    if use_pca and n_features > 50:
        print("  Applying PCA...")
        X_train, X_test, pca_model = extract_pca_features(X_train, X_test, n_components=0.95)

    results = train_and_evaluate_all(X_train, X_test, y_train, y_test, skip_models=skip_models)

    for model_name, result in results.items():
        if result.get('model') is None:
            print(f"\n{model_name}: FAILED — {result.get('error', 'unknown error')}")
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


def load_tabular_dataset(csv_path=None, tabular_dir=None):
    """Load tabular dataset (CSV) for dyslexia behavioral features."""
    import pandas as pd

    if csv_path:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    else:
        if not tabular_dir or not os.path.isdir(tabular_dir):
            raise FileNotFoundError(f"Tabular directory not found: {tabular_dir}")
        csv_files = [f for f in os.listdir(tabular_dir) if f.lower().endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {tabular_dir}")
        frames = []
        for f in csv_files:
            # Try semicolon first (Rello data), then comma
            path = os.path.join(tabular_dir, f)
            try:
                df_temp = pd.read_csv(path, sep=';', engine='python')
                if len(df_temp.columns) == 1:
                    df_temp = pd.read_csv(path, sep=None, engine='python')
            except Exception:
                df_temp = pd.read_csv(path, sep=None, engine='python')
            frames.append(df_temp)
        df = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {csv_files} ({len(df)} rows combined) from {tabular_dir}")

    target_col = None
    for col in df.columns:
        if col.lower() in ('dyslexia', 'label', 'target', 'class', 'diagnosis', 'risk'):
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[-1]
        print(f"Using last column '{target_col}' as target")

    y_raw = df[target_col]

    # --- LABEL ENCODING VALIDATION ---
    if not pd.api.types.is_numeric_dtype(y_raw):
        uniques = sorted(y_raw.dropna().unique().tolist())
        print(f"  Target '{target_col}' unique values: {uniques}")

        if len(uniques) != 2:
            raise ValueError(f"Expected a binary target in '{target_col}', found: {uniques}")

        # Try to auto-detect positive label
        positive_label = None
        for u in uniques:
            u_clean = str(u).strip().lower()
            if u_clean in ('yes', '1', 'true', 'dyslexia', 'dyslexic', 'high', 'risk'):
                positive_label = u
                break

        if positive_label is None:
            positive_label = uniques[-1]  # fallback to last

        y = (y_raw.astype(str).str.strip() == str(positive_label)).astype(int).values
        print(f"  Encoded target '{target_col}': '{positive_label}' -> 1, other -> 0")
        print(f"  Class distribution: {np.bincount(y)}")
    else:
        y = y_raw.values
        print(f"  Target '{target_col}' is numeric. Class distribution: {np.bincount(y)}")

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    feature_names = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    return X, y, feature_names


def train_dyslexia_aggregate(tabular_dir, output_dir):
    """Train aggregate-feature model from Rello visual search data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

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

    # Gender fix: handle 'M'/'F' or 'Male'/'Female' or 0/1
    gender_raw = data['Gender'].astype(str).str.strip().str.lower()
    agg['gender'] = gender_raw.map({'m': 1, 'male': 1, '1': 1, 'f': 0, 'female': 0, '0': 0}).fillna(0)

    # Verify gender mapping
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

    # Drop rows with all NaN
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
            print(f"\n{name}: FAILED — {result.get('error', 'unknown error')}")
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


def main():
    parser = argparse.ArgumentParser(description='Train dyslexia/dysgraphia detection models')
    parser.add_argument('--task', choices=['dysgraphia', 'dyslexia_tabular', 'dyslexia_handwriting', 
                                            'dyslexia_aggregate', 'all'],
                         default='all')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--csv-path', default=None)
    parser.add_argument('--max-images', type=int, default=None,
                         help='Hard cap per class (overrides auto-cap)')
    parser.add_argument('--full', action='store_true',
                         help='Disable auto-cap and skip-lists; train on ALL data with ALL models')
    parser.add_argument('--pca', action='store_true',
                         help='Apply PCA dimensionality reduction for image datasets')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    max_per_class = args.max_images
    if not args.full and max_per_class is None:
        max_per_class = DEFAULT_MAX_PER_CLASS
        print(f"Auto-capping classes at {max_per_class} images (use --full for unlimited)")
    elif args.full:
        max_per_class = None
        print("Full mode: no sampling caps, all models enabled")

    if args.task in ('dysgraphia', 'all'):
        train_image_dataset('dysgraphia', os.path.join(args.data_dir, 'dysgraphia'), args.output_dir,
                            max_images_per_class=max_per_class, use_pca=args.pca)

    if args.task in ('dyslexia_handwriting', 'all'):
        train_image_dataset('dyslexia_synthetic', os.path.join(args.data_dir, 'dyslexia_synthetic'), args.output_dir,
                            max_images_per_class=max_per_class, use_pca=args.pca)
        train_image_dataset('dyslexia_handwriting', os.path.join(args.data_dir, 'dyslexia_handwriting'), args.output_dir,
                            max_images_per_class=max_per_class, use_pca=args.pca)

    if args.task in ('dyslexia_tabular', 'all'):
        print("\n=== Training Dyslexia Tabular (Rello) Models ===")
        tabular_dir = os.path.join(args.data_dir, 'tabular')

        try:
            X, y, feature_names = load_tabular_dataset(csv_path=args.csv_path, tabular_dir=tabular_dir)
        except FileNotFoundError as e:
            print(f"  {e}")
            print(f"  Run prepare_dataset.py first, or pass --csv-path directly to a CSV.")
            X = None

        if X is not None:
            from sklearn.model_selection import train_test_split
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler

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
            joblib.dump(imputer, os.path.join(args.output_dir, 'dyslexia_tabular_imputer.pkl'))

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)

            joblib.dump(scaler, os.path.join(args.output_dir, 'dyslexia_tabular_scaler.pkl'))

            results = train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test)

            for name, result in results.items():
                if result.get('model') is None:
                    print(f"\n{name}: FAILED — {result.get('error', 'unknown error')}")
                    continue

                metrics = result['metrics']
                print(f"\n{name}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}")

                save_model(result['model'], os.path.join(args.output_dir, f'dyslexia_tabular_{name.lower().replace(" ", "_")}.pkl'))

            joblib.dump(feature_names, os.path.join(args.output_dir, 'dyslexia_tabular_feature_names.pkl'))
            print(f"\nDyslexia tabular models saved to {args.output_dir}")

    if args.task in ('dyslexia_aggregate', 'all'):
        train_dyslexia_aggregate(os.path.join(args.data_dir, 'tabular'), args.output_dir)


if __name__ == '__main__':
    main()