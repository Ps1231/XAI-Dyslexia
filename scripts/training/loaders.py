"""Dataset loaders for training: image folders and tabular CSVs."""
from __future__ import annotations

import os
import random

import numpy as np
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)

from scripts.common import console, IMG_SUFFIXES


def load_image_dataset(data_dir, max_images_per_class=None):
    """Load images from directory structure: data_dir/class_name/image.jpg.

    Returns (X_features, y_labels, feature_names).
    """
    import cv2
    from app.ml.feature_extraction import extract_all_features
    from app.ml.preprocessing import preprocess_pipeline

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
                  if f.lower().endswith(IMG_SUFFIXES)]

        if max_images_per_class and len(images) > max_images_per_class:
            print(f"  {class_name}: {len(images)} images -> sampling {max_images_per_class}")
            random.seed(42)
            images = random.sample(images, max_images_per_class)
        else:
            print(f"  {class_name}: {len(images)} images")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(f"[cyan]extracting [{class_name}]", total=len(images))
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    processed = preprocess_pipeline(img)
                    features, names = extract_all_features(processed)

                    if features is not None and len(features) > 0:
                        X_features.append(features)
                        y_labels.append(label_idx)
                        if not feature_names:
                            feature_names = names
                progress.advance(task_id)

    return np.array(X_features), np.array(y_labels), feature_names


def load_tabular_dataset(csv_path=None, tabular_dir=None):
    """Load tabular CSV(s) for dyslexia behavioral features.

    Auto-detects the target column and encodes binary labels
    ('yes'/'dyslexic'/1 -> 1). Returns (X, y, feature_names).
    """
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
