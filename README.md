# Lung Cancer Prediction Project (Flask + Tree Stack)

This project serves lung-cancer response and mortality predictions using a Flask UI and a tree-stack backend (LightGBM, XGBoost, TabNet).

## Current Architecture

- Frontend: `app.py`
- Service layer: `lungcancer/service.py`
- Feature schema: `lungcancer/schema.py`
- Training/evaluation: `lungcancer/tree_stack.py`
- Training runner: `lungcancer/train_tree_stack.py`
- Prepared datasets:
  - `data/processed/mortality_training_ready.csv`
  - `data/processed/response_training_ready.csv`

## Training

1. Install dependencies:

   `pip install -r requirements.txt`

2. Run all models for both tasks:

   `python -m lungcancer.train_tree_stack --task both --algorithm all`

3. Run with 150k mortality cap (faster, still comprehensive):

   `python -m lungcancer.train_tree_stack --task both --algorithm all --max-rows-mortality 150000`

## Outputs

Per `task` and `algorithm`, training writes:

- `data/models/<task>_<algorithm>_model.joblib`
- `data/models/<task>_<algorithm>_mapie.joblib` (optional)
- `data/reports/<task>_<algorithm>_nested_cv.json`
- `data/reports/<task>_<algorithm>_calibration_curve.png`
- `data/reports/<task>_<algorithm>_roc_curve.png`
- `data/reports/<task>_<algorithm>_probability_hist.png`
- `data/reports/<task>_<algorithm>_shap_summary.json` (LightGBM/XGBoost)
- `data/reports/<task>_<algorithm>_shap_summary.png` (LightGBM/XGBoost)

## Metrics In Reports

Each nested-CV report includes:

- `outer_fold_metrics` with `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `auc`, `brier`
- `outer_mean`, `outer_std`
- `outer_auc_ci_95` (95% CI for AUC from outer folds)
- `calibration_metrics`
- `best_threshold` and metrics at that threshold
- `artifacts` paths for model and plots

## Run the App

1. Install dependencies:

   `pip install -r requirements.txt`

2. Start Flask:

   `python app.py`

3. Open:

   `http://127.0.0.1:5000`

## Repository Structure

- `app.py`
- `lungcancer/`
  - `schema.py`
  - `service.py`
  - `harmonize_real_data.py`
  - `tree_stack.py`
  - `train_tree_stack.py`
- `data/`
  - `raw/`
  - `processed/`
  - `reports/`
  - `models/`
