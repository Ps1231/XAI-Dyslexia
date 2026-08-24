# XAI Dyslexia & Dysgraphia Screening System

> **Explainable AI for Early Detection of Dyslexia and Dysgraphia**
>
> A research-grade Flask application that combines classical machine learning with SHAP/LIME explainability to analyze handwriting samples and reading behavioral patterns for early screening of learning disabilities.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [One-Command Pipeline](#one-command-pipeline)
3. [Idempotency & Auto-Optimization](#idempotency--auto-optimization)
4. [Staged Usage](#staged-usage)
5. [Project Objectives](#project-objectives)
6. [System Architecture](#system-architecture)
7. [ML Pipeline](#ml-pipeline)
8. [Datasets Used](#datasets-used)
9. [Model Evaluation](#model-evaluation)
10. [Running the Application](#running-the-application)
11. [Running Tests](#running-tests)
12. [API Routes](#api-routes)
13. [Explainability Layer](#explainability-layer)
14. [Project Structure](#project-structure)
15. [Technologies](#technologies)
16. [Disclaimer](#disclaimer)

---

## Quickstart

```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EVERYTHING: download data -> train models -> evaluate
python -m scripts.pipeline

# 4. Launch the web app
python run.py                # http://127.0.0.1:5000
```

That's it. The pipeline is fully self-contained and re-runnable — safe to interrupt and resume at any time.

### Prerequisites

- Python 3.10+
- **Kaggle API token** (datasets 1–3): place `kaggle.json` at `~/.kaggle/kaggle.json` ([create one here](https://www.kaggle.com/settings) → API → Create New Token)
- **7-Zip or unrar** (dataset 3 ships as a password-protected RAR):
  - Windows: `winget install 7zip.7zip` (auto-detected from Program Files)
  - Linux: `sudo apt install p7zip-full`

---

## One-Command Pipeline

```bash
python -m scripts.pipeline
```

Runs all three stages in order:

| Stage | What it does | Output |
|-------|--------------|--------|
| **1. Prepare** | Downloads 5 public datasets, extracts nested/password archives, organizes into class folders | `data/processed/` |
| **2. Train** | Extracts ~1,787 features per image, trains 7 classifiers per task, saves artifacts | `app/models/*.pkl` |
| **3. Evaluate** | Computes accuracy/sensitivity/specificity/ROC-AUC, renders confusion matrices + ROC curves | `evaluation_plots/` |

Options:

```bash
python -m scripts.pipeline --task dysgraphia    # single task end-to-end
python -m scripts.pipeline --no-evaluate        # stop after training
python -m scripts.pipeline --force              # ignore markers, redo everything
```

Programmatic API:

```python
from scripts.pipeline import run_pipeline

run_pipeline()                    # everything, skipping finished work
run_pipeline(task="dysgraphia")
run_pipeline(evaluate=False)
run_pipeline(force=True)
```

---

## Idempotency & Auto-Optimization

The pipeline never repeats finished work:

| Mechanism | How |
|-----------|-----|
| **Dataset fingerprints** | Every dataset/model dir gets a SHA-256 over file paths+sizes (`.state/*.json` markers). Unchanged inputs → stage skipped instantly. |
| **Training skip** | If a task's data is unchanged AND its `.pkl` artifacts exist, training is skipped (`force=True` overrides). |
| **Evaluation skip** | Per-task fingerprint covers both the data and the exact model files; master report merges history so skips don't erase past results. |
| **Download skip** | Each source checks its folder before downloading; partial extractions resume where they stopped. |

Image budgets are **self-optimizing**:

```
dyslexia_handwriting: 47,584 images detected
  -> auto-cap: 125 images/class? no — budget math:
  -> 3 classes x 47k total > AUTO_TOTAL_BUDGET (6000)
  -> cap = 2000/class -> ~6,000 images actually used
```

- `AUTO_TOTAL_BUDGET = 6000` (training), `3000` (evaluation)
- Caps clamp between `MIN_CAP_PER_CLASS` / `MAX_CAP_PER_CLASS`
- Small datasets are used in full — capping only kicks in when needed
- Override with `max_images=N`, or disable entirely with `full=True`
- Slow SVMs auto-skip on large/high-dimensional datasets (RF, GB, MLP, LR, DT still train)

---

## Staged Usage

Each stage is also runnable standalone:

```bash
# Stage 1 only
python -m scripts.data.prepare [--force] [--skip-download] [--inspect]

# Stage 2 only
python -m scripts.training.train [--task TASK] [--full] [--pca] [--max-images N] [--force]

# Stage 3 only
python -m scripts.evaluation.evaluate [--task TASK] [--max-images N] [--force]
```

Every CLI is a thin shim over a programmatic function:

```python
from scripts.data.prepare import run_prepare
from scripts.training.train import run_training
from scripts.evaluation.evaluate import run_evaluation

run_prepare(skip_download=True)          # just organize what's downloaded
run_training(task="all", pca=True)
run_evaluation(task="dysgraphia", force=True)
```

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
│  │     /       │  │/dysgraphia  │  │  /dyslexia  │  │ /dyslexia/agg   │ │
│  │  Home       │  │  Upload     │  │  Reading    │  │ Visual Search   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
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
       │  Pipeline   │      │ Extraction  │      │ (sklearn)   │
       └─────────────┘      └─────────────┘      └──────┬──────┘
                                                        │
                              ┌─────────────────────────┼────────────────┐
                              │                         │                │
                       ┌──────▼──────┐          ┌──────▼──────┐  ┌──────▼──────┐
                       │    SHAP     │          │    LIME     │  │  Feature    │
                       │  (global)   │          │  (local)    │  │ Importance  │
                       └─────────────┘          └─────────────┘  └─────────────┘
```

---

## ML Pipeline

### Step 1: Data Collection
- 5 public datasets downloaded automatically by `scripts/data/prepare.py`
- Supports Kaggle (CLI), Zenodo (API), and Mendeley (direct download)

### Step 2: Preprocessing (Image)
```
Raw Image (BGR)
    → Grayscale
    → Gaussian Denoise
    → Otsu Binarization
    → Deskew (moment-based rotation correction)
    → Resize & Pad to 128×128
```

### Step 3: Feature Engineering (~1,787 features)
| Extractor | Features | Description |
|-----------|----------|-------------|
| HOG | ~1,764 | Histogram of Oriented Gradients (cell=8×8, block=2×2, 9 bins) |
| Contours | 4 | Aspect ratio mean, area variance, convex defects, perimeter/area ratio |
| Projections | 6 | Horizontal & vertical profile statistics (mean, std, max) |
| Morphology | 9 | Ink density, 7 Hu moments (log), stroke-width variance |
| Letter Spacing | 4 | Mean/std gap between components, baseline std & range |

### Step 4: Classification
| Model | Type | Notes |
|-------|------|-------|
| SVM (Linear) | Linear, calibrated | Fast baseline |
| SVM (RBF) | Non-linear, calibrated | Complex boundaries |
| Random Forest | Ensemble | Robust, interpretable |
| Gradient Boosting | Ensemble | High accuracy |
| MLP | Neural network | Non-linear patterns |
| Logistic Regression | Linear | Inherent interpretability |
| Decision Tree | Tree | Fully interpretable rules |

### Step 5: Explainability
- **SHAP**: game-theoretic feature attribution (TreeExplainer for trees, KernelExplainer otherwise)
- **LIME**: local linear approximation around each prediction
- **Feature Importance**: native importances (trees) or permutation importance (linear models)

---

## Datasets Used

| # | Dataset | Source | Type | Classes |
|---|---------|--------|------|---------|
| 1 | Synthetic Dyslexia Handwriting | Kaggle (michaelfink0923) | Image (YOLO labels) | Normal / Reversal / Corrected |
| 2 | Rello et al. Dyslexia | Kaggle (luzrello) | Tabular | Dyslexia (Yes/No) |
| 3 | Drizasazanitaisa Handwriting | Kaggle (drizasazanitaisa) | Image | Normal / Reversal / Corrected |
| 4 | Mendeley Dysgraphia | Mendeley (39hr8dx76p) | Image | Multiple class folders |
| 5 | ETDD70 Eye-Tracking | Zenodo (13332134) | Tabular + Gaze CSVs | Dyslexia labels |

Organized layout after Stage 1:

```
data/processed/
├── dysgraphia/<class>/            # dataset 4
├── dyslexia_synthetic/<class>/    # dataset 1 (YOLO label parsing)
├── dyslexia_handwriting/<class>/  # dataset 3
├── tabular/*.csv                  # dataset 2
├── eyetracking/*.csv              # dataset 5
└── .state/                        # pipeline fingerprints (auto-generated)
```

> **Note:** ETDD70 stimulus images are NOT used for image classification — the eye-tracking CSVs feed behavioral features instead.

---

## Model Evaluation

```bash
python -m scripts.evaluation.evaluate            # everything unfinished
python -m scripts.evaluation.evaluate --task dyslexia_aggregate --force
```

### Metrics Computed
- Accuracy, Precision, Recall, F1-Score
- Sensitivity / Specificity (binary tasks)
- ROC-AUC with rendered ROC curves
- Confusion matrix heatmaps + class distribution charts
- Full sklearn classification reports

### Output
```
evaluation_plots/
├── cm_<task>_<model>.png                  # confusion matrices
├── roc_<task>_<model>.png                 # ROC curves
├── <task>_class_distribution.png
├── master_evaluation_report.json          # merged history across runs
└── .state/                                # evaluation fingerprints
```

---

## Running the Application

```bash
python run.py                 # dev server -> http://127.0.0.1:5000
```

Production:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"
```

If models haven't been trained yet, upload pages show a clear message pointing to the training command.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

| Test File | What It Tests |
|-----------|---------------|
| `tests/test_preprocessing.py` | Pipeline output shape, deskew, binarize, denoise, empty image handling |
| `tests/test_feature_extraction.py` | Feature vector consistency, empty image handling, dimensionality |
| `tests/test_routes.py` | HTTP GET/POST for all routes, form validation, file upload, redirects |

---

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home page |
| `/about` | GET | About / methodology |
| `/dysgraphia/` | GET | Upload handwriting image |
| `/dysgraphia/analyze` | POST | Analyze uploaded image |
| `/dysgraphia/results` | GET | View analysis results |
| `/dyslexia/` | GET | Reading assessment form (5-feature model, heuristic fallback) |
| `/dyslexia/screen` | POST | Submit reading metrics |
| `/dyslexia/aggregate` | GET | Visual search form (8-feature aggregate model) |
| `/dyslexia/screen_aggregate` | POST | Submit aggregate metrics |
| `/dyslexia/results` | GET | View screening results |

Confidence handling: predictions below 60% confidence flag "Uncertain — Manual Review Recommended"; 60–85% shows "Possible indicators"; ≥85% gives a definitive label.

---

## Explainability Layer

Lives in `app/ml/explainability.py`:

| Function | Purpose |
|----------|---------|
| `get_shap_explanation()` | SHAP values — TreeExplainer (trees) or KernelExplainer (SVM/MLP/LR); returns None gracefully if shap isn't installed |
| `get_feature_importance()` | Native `feature_importances_` → absolute `coef_` → permutation importance |
| `get_lime_explanation()` | LimeTabularExplainer local weights around the instance |

Flow: prediction → explanations → JSON chart data (`generate_shap_plot_data()`, `generate_feature_importance_plot_data()`) → Chart.js waterfall + bar charts in the browser.

---

## Project Structure

```
XAI-Dyslexia/
│
├── app/
│   ├── __init__.py              # Flask app factory + blueprint registration
│   ├── config.py                # Config (MODEL_FOLDER, UPLOAD_FOLDER, etc.)
│   ├── ml/
│   │   ├── classifiers.py       # get_classifiers(), train_and_evaluate_all()
│   │   ├── explainability.py    # SHAP, LIME, feature importance wrappers
│   │   ├── feature_extraction.py # HOG, contours, projections, morphology, spacing
│   │   ├── predict.py           # predict_dysgraphia(), predict_dyslexia()
│   │   └── preprocessing.py     # load, grayscale, denoise, binarize, deskew, resize
│   ├── models/                  # Trained .pkl models (generated)
│   ├── routes/
│   │   ├── main.py              # Home + About
│   │   ├── dysgraphia.py        # Upload + analyze handwriting
│   │   └── dyslexia.py          # Reading form + aggregate form + results
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js           # Navbar, drag-drop upload, form validation
│   │   ├── js/charts.js         # Chart.js renderers (SHAP, importance, gauge)
│   │   └── uploads/             # User-uploaded images
│   └── templates/               # base, index, about, dysgraphia/, dyslexia/, partials/
│
├── data/
│   └── processed/               # Organized datasets (generated by Stage 1)
│
├── dyslexia_datasets/           # Raw downloads (generated by Stage 1)
│
├── scripts/
│   ├── __init__.py              # Path bootstrap for package execution
│   ├── common.py                # Fingerprints, state markers, fs helpers, constants
│   ├── pipeline.py              # Zero-config orchestrator (Stage 1→2→3)
│   ├── data/
│   │   ├── prepare.py           # run_prepare(): idempotent Stage 1
│   │   ├── sources.py           # The 5 dataset definitions + downloaders
│   │   ├── download.py          # HTTP + Kaggle CLI helpers (rich progress)
│   │   ├── extract.py           # zip/rar extraction (password-aware, Windows-safe)
│   │   ├── organize.py          # YOLO parsing + class-folder organization
│   │   └── inspect_data.py      # Human-readable dataset summaries
│   ├── training/
│   │   ├── train.py             # run_training(): idempotent Stage 2
│   │   ├── trainer.py           # Task trainers, dynamic auto-cap, up-to-date guards
│   │   └── loaders.py           # Image-folder + tabular CSV loaders (rich progress)
│   └── evaluation/
│       ├── evaluate.py          # run_evaluation(): idempotent Stage 3
│       ├── tasks.py             # Per-task evaluators (image + tabular)
│       └── plots.py             # Confusion matrix, ROC, distribution charts
│
├── tests/                       # unittest/pytest suites
├── notebooks/                   # Exploration notebooks
├── evaluation_plots/            # Generated by Stage 3
├── requirements.txt
├── run.py                       # Flask entry point
└── README.md
```

---

## Technologies

| Layer | Stack |
|-------|-------|
| **Backend** | Python 3.10+, Flask 3.x |
| **ML** | Scikit-learn, NumPy, SciPy, XGBoost |
| **Image Processing** | OpenCV, Pillow |
| **Explainability** | SHAP, LIME (optional, graceful fallback) |
| **Frontend** | HTML5, CSS3, Vanilla JS, Chart.js |
| **Data** | Pandas, Requests, Kaggle API, Zenodo API |
| **UX** | Rich (progress bars, console output) |
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
