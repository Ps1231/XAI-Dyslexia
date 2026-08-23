# XAI Dyslexia & Dysgraphia Screening System

> **Explainable AI for Early Detection of Dyslexia and Dysgraphia**
>
> A research-grade Flask application that combines classical machine learning with SHAP/LIME explainability to analyze handwriting samples and reading behavioral patterns for early screening of learning disabilities.

---

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [System Architecture](#system-architecture)
3. [ML Pipeline](#ml-pipeline)
4. [Datasets Used](#datasets-used)
5. [Setup & Installation](#setup--installation)
6. [Data Preparation](#data-preparation)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Running the Application](#running-the-application)
10. [Running Tests](#running-tests)
11. [API Routes](#api-routes)
12. [Explainability Layer](#explainability-layer)
13. [Project Structure](#project-structure)
14. [Technologies](#technologies)
15. [Disclaimer](#disclaimer)

---

## Project Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | **Early Detection** — Identify potential indicators of dyslexia and dysgraphia in children at an early age | ✅ Implemented |
| 2 | **Explainable Predictions** — SHAP and LIME-based explanations for every prediction | ✅ Implemented |
| 3 | **Multi-Modal Analysis** — Handwriting images (dysgraphia) + reading behavioral data (dyslexia) | ✅ Implemented |
| 4 | **Accessible Tools** — Web-based interface deployable in schools, clinics, and homes | ✅ Implemented |
| 5 | **Comprehensive Evaluation** — Accuracy, sensitivity, specificity, ROC-AUC, confusion matrices | ✅ Implemented |
| 6 | **Automated Data Pipeline** — Download, extract, and organize 5 public datasets automatically | ✅ Implemented |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FLASK WEB APP                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   /         │  │/dysgraphia  │  │  /dyslexia  │  │ /dyslexia/agg   │ │
│  │  Home       │  │  Upload     │  │  Reading    │  │ Visual Search   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
│         │                │                │                  │          │
│         └────────────────┴────────────────┴──────────────────┘          │
│                                    │                                    │
│                         ┌──────────┴──────────┐                         │
│                         │   ML PREDICTION     │                         │
│                         │   + EXPLAINABILITY  │                         │
│                         └──────────┬──────────┘                         │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
       ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
       │ Preprocess  │      │  Feature    │      │  Classifier │
       │  Pipeline   │      │ Extraction  │      │   (sklearn) │
       │             │      │             │      │             │
       │ • Grayscale │      │ • HOG       │      │ • SVM       │
       │ • Denoise   │      │ • Contours  │      │ • RandomF   │
       │ • Binarize  │      │ • Projections│     │ • GradBoost │
       │ • Deskew    │      │ • Morphology│      │ • MLP       │
       │ • Resize    │      │ • Spacing   │      │ • LogReg    │
       └─────────────┘      └─────────────┘      └──────┬──────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         │                         │
                       ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
                       │    SHAP     │          │    LIME     │          │  Feature    │
                       │  (global)   │          │  (local)    │          │ Importance  │
                       └─────────────┘          └─────────────┘          └─────────────┘
```

---

## ML Pipeline

### Step 1: Data Collection
- 5 public datasets downloaded automatically via `prepare_dataset.py`
- Supports Kaggle, Zenodo, and Mendeley sources

### Step 2: Preprocessing (Image)
```
Raw Image (BGR)
    → Grayscale
    → Gaussian Denoise
    → Otsu Binarization
    → Deskew (moment-based rotation correction)
    → Resize & Pad to 128×128
```

### Step 3: Feature Engineering
| Extractor | Features | Description |
|-----------|----------|-------------|
| **HOG** | ~1,764 | Histogram of Oriented Gradients (cell=8×8, block=2×2, 9 bins) |
| **Contours** | 4 | Aspect ratio mean, area variance, convex defects, perimeter/area ratio |
| **Projections** | 6 | Horizontal & vertical profile statistics (mean, std, max) |
| **Morphology** | 9 | Ink density, 7 Hu moments (log), stroke-width variance |
| **Letter Spacing** | 4 | Mean/std gap between components, baseline std & range |
| **Total** | ~1,787 | Concatenated into single feature vector |

### Step 4: Classification
| Model | Type | Best For |
|-------|------|----------|
| SVM (Linear) | Linear kernel | Fast baseline |
| SVM (RBF) | Non-linear kernel | Complex boundaries |
| Random Forest | Ensemble | Robust, interpretable |
| Gradient Boosting | Ensemble | High accuracy |
| MLP | Neural network | Non-linear patterns |
| Logistic Regression | Linear | Inherent interpretability |
| Decision Tree | Tree | Fully interpretable rules |

### Step 5: Explainability
- **SHAP**: Game-theoretic feature attribution showing how each feature pushes the prediction
- **LIME**: Local linear approximation around the prediction instance
- **Feature Importance**: Native model importances (trees) or permutation importance (linear models)

---

## Datasets Used

| # | Dataset | Source | Type | Classes |
|---|---------|--------|------|---------|
| 1 | Synthetic Dyslexia Handwriting | Kaggle (michaelfink0923) | Image (YOLO labels) | Normal / Reversal / Corrected |
| 2 | Rello et al. Dyslexia | Kaggle (luzrello) | Tabular | Dyslexia (Yes/No) |
| 3 | Drizasazanitaisa Handwriting | Kaggle (drizasazanitaisa) | Image | Normal / Reversal / Corrected |
| 4 | Mendeley Dysgraphia | Mendeley (39hr8dx76p) | Image | Multiple class folders |
| 5 | ETDD70 Eye-Tracking | Zenodo (13332134) | Tabular + Gaze | Dyslexia labels |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- `unzip`, `unrar` or `7z` (for dataset extraction)
- Kaggle API credentials (for Kaggle datasets)

### 1. Clone & Create Virtual Environment
```bash
git clone <repo-url>
cd XAI-Dyslexia
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
flask>=2.3
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
opencv-python>=4.8
scipy>=1.11
joblib>=1.3
matplotlib>=3.7
chart.js (CDN, no pip)
# Optional but recommended:
shap>=0.42
lime>=0.2
```

### 3. Configure Kaggle (if downloading datasets)
```bash
pip install kaggle
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Data Preparation

### Download & Organize All Datasets
```bash
python scripts/prepare_dataset.py
```

This will:
1. Download all 5 datasets to `dyslexia_datasets/`
2. Extract archives (handles nested zips, password-protected rars, macOS junk)
3. Organize into `data/processed/` with structure:
```
data/processed/
├── dysgraphia/
│   ├── class_a/
│   └── class_b/
├── dyslexia_synthetic/
│   ├── normal/
│   └── dyslexic/
├── dyslexia_handwriting/
│   ├── normal/
│   └── corrected/
├── tabular/
│   └── *.csv
└── eyetracking/
    └── *.csv
```

### Skip Download (use existing)
```bash
python scripts/prepare_dataset.py --skip-download
```

---

## Model Training

### Train All Models
```bash
python scripts/train_models.py --task all
```

### Train Specific Task
```bash
python scripts/train_models.py --task dysgraphia
python scripts/train_models.py --task dyslexia_tabular
python scripts/train_models.py --task dyslexia_handwriting
python scripts/train_models.py --task dyslexia_aggregate
```

### Training Options
| Flag | Description |
|------|-------------|
| `--full` | No sampling cap, train ALL models including slow SVMs |
| `--max-images N` | Cap images per class (default: 2000) |
| `--pca` | Apply PCA dimensionality reduction |
| `--data-dir PATH` | Custom data directory |
| `--output-dir PATH` | Custom model output directory |

### Output
Trained models saved to `app/models/`:
```
app/models/
├── dysgraphia_random_forest.pkl
├── dysgraphia_gradient_boosting.pkl
├── dysgraphia_mlp.pkl
├── dyslexia_tabular_random_forest.pkl
├── dyslexia_tabular_scaler.pkl
├── dyslexia_tabular_imputer.pkl
├── dyslexia_aggregate_random_forest.pkl
├── dyslexia_aggregate_scaler.pkl
└── ...
```

---

## Model Evaluation

### Evaluate All Models
```bash
python scripts/evaluate.py --task all
```

### Evaluate Specific Task
```bash
python scripts/evaluate.py --task dyslexia_aggregate
python scripts/evaluate.py --task dysgraphia
```

### Metrics Computed
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **Sensitivity** (True Positive Rate)
- **Specificity** (True Negative Rate)
- **ROC-AUC** (with ROC curve plots)
- **Confusion Matrix** (visual heatmap)
- **Class Distribution**

### Output
```
evaluation_plots/
├── roc_aggregate_random_forest.png
├── cm_aggregate_random_forest.png
├── aggregate_evaluation_report.json
├── dysgraphia_evaluation_report.json
└── ...
```

---

## Running the Application

### Development Server
```bash
python run.py
```
App runs at `http://127.0.0.1:5000`

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"
```

### Application Routes
| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page |
| `/about` | GET | About / methodology |
| `/dysgraphia/` | GET | Upload handwriting image |
| `/dysgraphia/analyze` | POST | Analyze uploaded image |
| `/dysgraphia/results` | GET | View analysis results |
| `/dyslexia/` | GET | Reading assessment form |
| `/dyslexia/screen` | POST | Submit reading metrics |
| `/dyslexia/aggregate` | GET | Visual search form |
| `/dyslexia/screen_aggregate` | POST | Submit aggregate metrics |
| `/dyslexia/results` | GET | View screening results |

---

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Individual Test Files
```bash
python tests/test_preprocessing.py
python tests/test_feature_extraction.py
python tests/test_routes.py
```

### Test Coverage
| Test File | What It Tests |
|-----------|---------------|
| `test_preprocessing.py` | Pipeline output shape, deskew, binarize, denoise, empty image handling |
| `test_feature_extraction.py` | Feature vector consistency, empty image handling, dimensionality checks |
| `test_routes.py` | HTTP GET/POST for all routes, form validation, file upload, redirects |

---

## Explainability Layer

### Where It Lives
```
app/ml/explainability.py
```

### How It Flows
```
User Input
    → predict_dysgraphia() / predict_dyslexia()
        → get_shap_explanation()      → SHAP values array
        → get_feature_importance()   → {feature: score} dict
        → get_lime_explanation()     → Local linear weights
    → generate_shap_plot_data()      → JSON for Chart.js waterfall
    → generate_feature_importance_plot_data() → JSON for horizontal bar chart
    → Frontend (results.html)
        → Chart.js renders SHAP waterfall + Feature importance bars
```

### What Each Method Does

**`get_shap_explanation(model, X, feature_names, model_type)`**
- Uses `shap.TreeExplainer` for tree-based models (Random Forest, GB, DT)
- Falls back to `shap.KernelExplainer` for model-agnostic explanations (SVM, MLP, LR)
- Returns SHAP values array shaped for multi-class handling

**`get_feature_importance(model, feature_names, X, y)`**
- Priority 1: Native `feature_importances_` (tree models)
- Priority 2: Absolute `coef_` (linear models)
- Priority 3: Permutation importance (requires X, y validation data)
- Returns sorted `{feature_name: importance_score}` dictionary

**`get_lime_explanation(model, X_train, instance, feature_names, class_names)`**
- Fits `LimeTabularExplainer` on training data distribution
- Perturbs the single instance and fits a local linear model
- Returns prediction, local feature weights, and intercept

**Shape Guards in `generate_shap_plot_data()`**
- Handles 0-D, 2-D, and 3-D SHAP outputs safely
- Correctly indexes `(n_classes, n_samples, n_features)` vs `(n_samples, n_features, n_classes)`
- Prevents index-out-of-bounds crashes on edge-case model outputs

---

## Project Structure

```
XAI-Dyslexia/
│
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration (MODEL_FOLDER, UPLOAD_FOLDER, etc.)
│   ├── ml/
│   │   ├── classifiers.py       # get_classifiers(), train_and_evaluate_all()
│   │   ├── explainability.py    # SHAP, LIME, feature importance wrappers
│   │   ├── feature_extraction.py # HOG, contours, projections, morphology, spacing
│   │   ├── predict.py           # predict_dysgraphia(), predict_dyslexia()
│   │   └── preprocessing.py   # load, grayscale, denoise, binarize, deskew, resize
│   ├── models/                  # Trained .pkl models (generated by train_models.py)
│   ├── routes/
│   │   ├── dysgraphia.py        # Upload + analyze handwriting
│   │   ├── dyslexia.py          # Reading form + aggregate form + results
│   │   └── main.py              # Home + About
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js           # Navbar, drag-drop upload, form validation
│   │   ├── js/charts.js         # Chart.js renderers (SHAP, importance, gauge)
│   │   └── uploads/             # User-uploaded images
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── about.html
│       ├── dysgraphia/
│       │   ├── upload.html
│       │   └── results.html
│       ├── dyslexia/
│       │   ├── test.html
│       │   ├── test_aggregate.html
│       │   └── results.html
│       └── partials/
│           ├── _navigation.html
│           ├── _result_card.html
│           └── _upload_form.html
│
├── data/
│   └── processed/               # Organized datasets
│
├── dyslexia_datasets/           # Raw downloaded datasets
│
├── scripts/
│   ├── prepare_dataset.py       # Download + extract + organize 5 datasets
│   ├── train_models.py          # Train all classifiers
│   └── evaluate.py              # Comprehensive evaluation + plots
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_routes.py
│
├── evaluation_plots/            # Generated by evaluate.py
│
├── requirements.txt
├── run.py                       # Entry point
└── README.md
```

---

## Technologies

| Layer | Stack |
|-------|-------|
| **Backend** | Python 3.10+, Flask |
| **ML/DL** | Scikit-learn, NumPy, SciPy |
| **Image Processing** | OpenCV, Pillow |
| **Explainability** | SHAP, LIME |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, Chart.js |
| **Data** | Pandas, CSV, Kaggle API, Zenodo API |
| **Testing** | unittest, pytest |
| **Deployment** | Flask dev server / Gunicorn |

---

## Disclaimer

> **This tool is designed for screening and educational purposes only.**
>
> It is **not a diagnostic instrument** and should not be used as a substitute for professional clinical assessment. Always consult qualified healthcare professionals or educational psychologists for formal diagnosis of learning disabilities.
>
> The models were trained on limited public datasets and may not generalize to all populations, languages, handwriting styles, or age groups. Results are intended for demonstration and research purposes.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{xai_dyslexia_2026,
  title = {Explainable AI for Dyslexia and Dysgraphia Screening},
  year = {2026},
  note = {Research project combining SHAP/LIME explainability with classical ML}
}
```

---

## License

MIT License — For educational and research purposes.
