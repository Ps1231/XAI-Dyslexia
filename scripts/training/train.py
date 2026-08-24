"""Stage 2 — train all models (idempotent, dynamic auto-capping).

Programmatic:
    from scripts.training.train import run_training
    run_training()                       # everything unfinished
    run_training(task="dysgraphia")
    run_training(task="all", force=True) # ignore up-to-date markers

CLI:
    python -m scripts.training.train [--task TASK] [--full] [--pca] ...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np

from scripts.common import dir_fingerprint, load_state, save_state
from scripts.training.loaders import load_tabular_dataset
from scripts.training.trainer import (
    _mark_task_done,
    _task_up_to_date,
    train_dyslexia_aggregate,
    train_image_dataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "app" / "models"

VALID_TASKS = ("dysgraphia", "dyslexia_tabular", "dyslexia_handwriting",
               "dyslexia_aggregate", "all")


def run_training(
    task: str = "all",
    data_dir=None,
    output_dir=None,
    csv_path=None,
    max_images: int | str | None = None,
    full: bool = False,
    pca: bool = False,
    force: bool = False,
):
    """Train models for one or more tasks — skipping anything already trained.

    Args:
        task: One of VALID_TASKS.
        data_dir: Directory containing organized datasets
            (default <project>/data/processed).
        output_dir: Where trained .pkl models are saved
            (default <project>/app/models).
        csv_path: Optional direct path to a tabular CSV (overrides data_dir/tabular).
        max_images: Per-class image cap. None (default) = dynamic auto-cap that
            keeps total images within AUTO_TOTAL_BUDGET; pass an int to pin it;
            ignored when full=True.
        full: Disable caps and skip-lists; train on ALL data with ALL models.
        pca: Apply PCA dimensionality reduction for image datasets.
        force: Retrain even when the data is unchanged since last run.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {', '.join(VALID_TASKS)}")

    data_dir = str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR)
    output_dir = str(Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR)

    os.makedirs(output_dir, exist_ok=True)

    if full:
        max_per_class = None
        print("Full mode: no sampling caps, all models enabled")
    elif max_images is not None:
        max_per_class = max_images
        print(f"Using explicit cap: {max_images} images/class")
    else:
        max_per_class = "auto"

    if task in ('dysgraphia', 'all'):
        src = os.path.join(data_dir, 'dysgraphia')
        if not _task_up_to_date(src, output_dir, 'dysgraphia', force):
            train_image_dataset('dysgraphia', src, output_dir,
                                max_images_per_class=max_per_class, use_pca=pca)
            _mark_task_done(src, output_dir, 'dysgraphia')

    if task in ('dyslexia_handwriting', 'all'):
        for name in ('dyslexia_synthetic', 'dyslexia_handwriting'):
            src = os.path.join(data_dir, name)
            if not _task_up_to_date(src, output_dir, name, force):
                train_image_dataset(name, src, output_dir,
                                    max_images_per_class=max_per_class, use_pca=pca)
                _mark_task_done(src, output_dir, name)

    if task in ('dyslexia_tabular', 'all'):
        tabular_dir = os.path.join(data_dir, 'tabular')
        if not _task_up_to_date(tabular_dir, output_dir, 'dyslexia_tabular', force):
            print("\n=== Training Dyslexia Tabular (Rello) Models ===")

            X = None
            try:
                X, y, feature_names = load_tabular_dataset(csv_path=csv_path, tabular_dir=tabular_dir)
            except FileNotFoundError as e:
                print(f"  {e}")
                print(f"  Run prepare first, or pass csv_path directly to a CSV.")

            if X is not None:
                from sklearn.model_selection import train_test_split
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from app.ml.classifiers import train_and_evaluate_all, save_model

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
                joblib.dump(imputer, os.path.join(output_dir, 'dyslexia_tabular_imputer.pkl'))

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed)
                X_test_scaled = scaler.transform(X_test_imputed)

                joblib.dump(scaler, os.path.join(output_dir, 'dyslexia_tabular_scaler.pkl'))

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

                    save_model(result['model'], os.path.join(
                        output_dir, f'dyslexia_tabular_{name.lower().replace(" ", "_")}.pkl'))

                joblib.dump(feature_names, os.path.join(output_dir, 'dyslexia_tabular_feature_names.pkl'))
                print(f"\nDyslexia tabular models saved to {output_dir}")

            if X is not None:
                _mark_task_done(tabular_dir, output_dir, 'dyslexia_tabular')

    if task in ('dyslexia_aggregate', 'all'):
        src = os.path.join(data_dir, 'tabular')
        if not _task_up_to_date(src, output_dir, 'dyslexia_aggregate', force):
            train_dyslexia_aggregate(src, output_dir)
            _mark_task_done(src, output_dir, 'dyslexia_aggregate')


def main():
    """Optional CLI shim — every flag maps to a run_training() kwarg."""
    parser = argparse.ArgumentParser(description='Train dyslexia/dysgraphia detection models')
    parser.add_argument('--task', choices=list(VALID_TASKS), default='all')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--csv-path', default=None)
    parser.add_argument('--max-images', type=int, default=None,
                        help='Hard cap per class (default: dynamic auto-cap)')
    parser.add_argument('--full', action='store_true',
                        help='Disable auto-cap and skip-lists; train on ALL data with ALL models')
    parser.add_argument('--pca', action='store_true',
                        help='Apply PCA dimensionality reduction for image datasets')
    parser.add_argument('--force', action='store_true',
                        help='Retrain even when nothing changed')
    args = parser.parse_args()

    run_training(
        task=args.task,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        max_images=args.max_images,
        full=args.full,
        pca=args.pca,
        force=args.force,
    )


if __name__ == '__main__':
    main()
