"""
Train and save ML models for dysgraphia and dyslexia detection.

Usage:
    python scripts/train_models.py --data-dir data/processed --output-dir app/models
"""
import os
import sys
import argparse
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.classifiers import get_classifiers, train_and_evaluate_all, save_model
from app.ml.feature_extraction import extract_all_features
from app.ml.preprocessing import preprocess_pipeline
import cv2


def load_image_dataset(data_dir):
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
        
        print(f"  {class_name}: {len(images)} images")
        
        for img_name in images:
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


def load_tabular_dataset(csv_path):
    """Load tabular dataset (CSV) for dyslexia behavioral features."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    target_col = None
    for col in df.columns:
        if col.lower() in ('dyslexia', 'label', 'target', 'class', 'diagnosis', 'risk'):
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[-1]
        print(f"Using last column '{target_col}' as target")
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = [c for c in df.columns if c != target_col]
    
    return X, y, feature_names


def main():
    parser = argparse.ArgumentParser(description='Train dyslexia/dysgraphia detection models')
    parser.add_argument('--task', choices=['dysgraphia', 'dyslexia', 'all'], default='all')
    parser.add_argument('--data-dir', default='data/processed')
    parser.add_argument('--output-dir', default='app/models')
    parser.add_argument('--csv-path', default=None, help='Path to tabular CSV for dyslexia')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task in ('dysgraphia', 'all'):
        print("\n=== Training Dysgraphia Models ===")
        dysgraphia_dir = os.path.join(args.data_dir, 'dysgraphia')
        
        if os.path.exists(dysgraphia_dir):
            X, y, feature_names = load_image_dataset(dysgraphia_dir)
            if X is not None and len(X) > 0:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                results = train_and_evaluate_all(X_train, X_test, y_train, y_test)
                
                for name, result in results.items():
                    print(f"\n{name}:")
                    print(f"  Accuracy:  {result['accuracy']:.4f}")
                    print(f"  Precision: {result['precision']:.4f}")
                    print(f"  Recall:    {result['recall']:.4f}")
                    print(f"  F1 Score:  {result['f1']:.4f}")
                    
                    model = result.get('model')
                    if model is not None:
                        save_model(model, os.path.join(args.output_dir, f'dysgraphia_{name.lower().replace(" ", "_")}.pkl'))
                
                joblib.dump(feature_names, os.path.join(args.output_dir, 'dysgraphia_feature_names.pkl'))
                print(f"\nDysgraphia models saved to {args.output_dir}")
            else:
                print("No valid images found for dysgraphia training")
        else:
            print(f"Dysgraphia data directory not found: {dysgraphia_dir}")
    
    if args.task in ('dyslexia', 'all'):
        print("\n=== Training Dyslexia Models ===")
        csv_path = args.csv_path or os.path.join(args.data_dir, 'dyslexia_features.csv')
        
        if os.path.exists(csv_path):
            X, y, feature_names = load_tabular_dataset(csv_path)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            joblib.dump(scaler, os.path.join(args.output_dir, 'dyslexia_scaler.pkl'))
            
            results = train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test)
            
            for name, result in results.items():
                print(f"\n{name}:")
                print(f"  Accuracy:  {result['accuracy']:.4f}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall:    {result['recall']:.4f}")
                print(f"  F1 Score:  {result['f1']:.4f}")
                
                model = result.get('model')
                if model is not None:
                    save_model(model, os.path.join(args.output_dir, f'dyslexia_{name.lower().replace(" ", "_")}.pkl'))
            
            joblib.dump(feature_names, os.path.join(args.output_dir, 'dyslexia_feature_names.pkl'))
            print(f"\nDyslexia models saved to {args.output_dir}")
        else:
            print(f"Dyslexia CSV not found: {csv_path}")
            print("Place your dataset CSV in the data/processed/ directory")


if __name__ == '__main__':
    main()
