# Tasks

## Project Setup
- [x] Create README with dataset links
- [x] Create project directory structure
- [x] Create tasks.md
- [x] Create requirements.txt
- [x] Create run.py entry point
- [x] Create .gitignore
- [x] Create config.py

## ML Pipeline
- [x] Build preprocessing.py (grayscale, threshold, deskew, noise removal)
- [x] Build feature_extraction.py (HOG, contours, projections, morphological)
- [x] Build classifiers.py (SVM, Random Forest, XGBoost training & evaluation)
- [x] Build explainability.py (SHAP, LIME, feature importance)
- [x] Build predict.py (inference + explanation generation)

## Flask Application
- [x] Create app factory (__init__.py)
- [x] Create main routes (homepage, about)
- [x] Create dysgraphia routes (upload, analyze, results)
- [x] Create dyslexia routes (reading test, results)
- [x] Create base.html + index.html templates
- [x] Create dysgraphia upload + results templates
- [x] Create dyslexia test + results templates
- [x] Create CSS stylesheet
- [x] Create JS for upload, charts, SHAP visualizations

## Scripts
- [x] Create train_models.py (CLI to train & save all models)
- [x] Create prepare_dataset.py (download & preprocess datasets)

## XAI / Explainability
- [x] SHAP summary plots for global feature importance
- [x] Per-prediction SHAP waterfall plots
- [x] LIME explanations for individual classifications
- [x] Feature importance bar charts on results page

## Testing
- [ ] Unit tests for preprocessing
- [ ] Unit tests for feature extraction
- [ ] Unit tests for routes

## Final
- [ ] Install dependencies and verify app runs
- [ ] Final README polish
