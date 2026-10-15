# Heart Disease Detection

A reproducible cardiovascular disease risk classification pipeline using a hybrid ensemble of Random Forest and SVM models. Built for research and educational purposes using the UCI Heart Disease (Cleveland) dataset.

> **Disclaimer:** This project is for research and educational purposes only. Predictions are not a substitute for professional medical advice. This system has not been clinically validated.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Reproducing Results](#reproducing-results)
5. [Feature Definitions](#feature-definitions)
6. [Model Architecture](#model-architecture)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Prediction Service](#prediction-service)
9. [Limitations](#limitations)

---

## Project Structure

```
heart-disease-detection/
├── data/
│   └── README.md              # Dataset attribution, schema, target mapping
├── artifacts/                 # Saved model pipelines (gitignored)
│   └── .gitkeep
├── src/
│   └── heart_disease/
│       ├── __init__.py
│       ├── config.py          # All constants: paths, seeds, hyperparameter grids
│       ├── data_loader.py     # UCI fetch, validation, binary target mapping, EDA
│       ├── preprocessing.py   # ColumnTransformer pipeline (imputation + scaling)
│       ├── train.py           # RF and SVM training with GridSearchCV
│       ├── evaluate.py        # Metrics, plots, comparison table
│       ├── ensemble.py        # Weighted fusion + EnsemblePredictor
│       ├── predict.py         # Inference service (loads artifacts, validates, predicts)
│       ├── schemas.py         # Input/output dataclasses with validation
│       └── run_pipeline.py    # End-to-end training + evaluation entry point
├── tests/
│   ├── test_preprocessing.py  # Pipeline build, NaN imputation, no-leakage checks
│   ├── test_ensemble.py       # Fusion formula, weight selection, persistence
│   └── test_prediction.py     # Input validation, serialisation, integration test
├── reports/                   # Auto-generated plots and CSV (gitignored)
│   └── .gitkeep
├── conftest.py                # Pytest sys.path configuration
├── app.py                     # Interactive Streamlit frontend dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Name | UCI Heart Disease |
| ID | 45 |
| URL | https://archive.ics.uci.edu/dataset/45/heart+disease |
| Subset | Cleveland only (the only one used in published ML research) |
| Instances | 303 |
| Input features | 13 |
| License | CC BY 4.0 |
| Paper | Detrano et al., *American Journal of Cardiology*, 1989 |

**Binary target mapping:**

| Raw `num` | Label | Meaning |
|---|---|---|
| 0 | 0 | No disease — < 50% diameter narrowing |
| 1, 2, 3, 4 | 1 | Disease present — ≥ 50% narrowing in ≥ 1 vessel |

The dataset is fetched automatically at runtime via `ucimlrepo`. **No manual download is required.**

---

## Setup

### Prerequisites
- Python 3.10 or later
- Internet access (for first-time dataset download via `ucimlrepo`)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd heart-disease-detection

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Results

All results are reproducible with `RANDOM_SEED = 42` (set in `config.py`).

### Run the full pipeline

```bash
PYTHONPATH=src python -m heart_disease.run_pipeline
```

This will:
1. Download the Cleveland dataset via `ucimlrepo`
2. Print an EDA summary
3. Train Random Forest and SVM with `GridSearchCV`
4. Select the fusion weight via cross-validation on the training set
5. Evaluate all three models on the held-out test set
6. Save reports to `reports/` and artifacts to `artifacts/`

### Run tests only

```bash
PYTHONPATH=src pytest tests/ -v
```

The unit tests (preprocessing, ensemble, validation) run entirely on synthetic data and do **not** require the trained artifacts. The integration test in `test_prediction.py` is automatically skipped if artifacts are absent.

### Launch the Streamlit Frontend Web App

```bash
streamlit run app.py
```

This starts the interactive web dashboard on `http://localhost:8501`, featuring:
- **Patient Risk Assessment Form** with demographic, vitals, stress test, and fluoroscopy inputs.
- **One-click presets** for healthy and high-risk cases.
- **Instant multi-model predictions**: RF, SVM, and Weighted Hybrid consensus.
- **Visual risk gauges and probability distribution charts**.
- **Model Benchmarks & Research dashboard** showing test metrics, ROC curves, confusion matrices, and calibration curves.

### Outputs after running the pipeline

| File | Contents |
|---|---|
| `artifacts/rf_pipeline.joblib` | Fitted Random Forest pipeline |
| `artifacts/svm_pipeline.joblib` | Fitted SVM pipeline |
| `artifacts/ensemble_meta.joblib` | EnsemblePredictor (w, threshold, both pipelines) |
| `reports/comparison_table.csv` | Accuracy/Precision/Recall/Specificity/F1/AUC for all models |
| `reports/confusion_matrices.png` | Side-by-side confusion matrices |
| `reports/roc_curves.png` | Overlaid ROC curves |
| `reports/calibration_curves.png` | Reliability diagrams |
| `reports/run_metadata.json` | Seed, best params, fusion weight, CV AUC |

---

## Feature Definitions

| Feature | Type | Description | Units | Valid values |
|---|---|---|---|---|
| `age` | Integer | Age in years | years | 1–120 |
| `sex` | Binary | Patient sex | — | 0 (female), 1 (male) |
| `cp` | Categorical | Chest pain type | — | 1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic |
| `trestbps` | Integer | Resting blood pressure (on admission) | mm Hg | 50–300 |
| `chol` | Integer | Serum cholesterol | mg/dl | 50–700 |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl | — | 0 (false), 1 (true) |
| `restecg` | Categorical | Resting ECG results | — | 0=normal, 1=ST-T abnormality, 2=LV hypertrophy |
| `thalach` | Integer | Maximum heart rate achieved | bpm | 40–280 |
| `exang` | Binary | Exercise-induced angina | — | 0 (no), 1 (yes) |
| `oldpeak` | Float | ST depression induced by exercise relative to rest | — | -5 to 10 |
| `slope` | Categorical | Slope of peak exercise ST segment | — | 1=upsloping, 2=flat, 3=downsloping |
| `ca` | Integer | Major vessels coloured by fluoroscopy | — | 0–3 |
| `thal` | Categorical | Thalassemia | — | 3=normal, 6=fixed defect, 7=reversable defect |

Missing values in `ca` and `thal` are imputed using the most-frequent value (mode) of the **training set only**.

---

## Model Architecture

### Preprocessing

```
Numeric  [age, trestbps, chol, thalach, oldpeak]
  → SimpleImputer(median) → StandardScaler()

Categorical  [sex, cp, fbs, restecg, exang, slope, ca, thal]
  → SimpleImputer(most_frequent) → OrdinalEncoder   [for RF]
                                 → OneHotEncoder    [for SVM]
```

All transformers are fitted **exclusively on training data** to prevent leakage.

### Random Forest
- `RandomForestClassifier(class_weight="balanced", random_state=42)`
- Grid search: `{max_depth, min_samples_leaf, n_estimators}` via 5-fold stratified CV

### SVM
- `SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)`
- Grid search: `{C, gamma}` via 5-fold stratified CV

### Hybrid Ensemble

```
P_hybrid = w × P_RF + (1 - w) × P_SVM
```

- `w` selected by maximising mean ROC-AUC on **cross-validation folds of the training set only**
- Default decision threshold: `0.50`
- See `reports/run_metadata.json` for the chosen `w` value

---

## Evaluation Methodology

- **Split:** Stratified 80/20 train/test (`random_state=42`)
- **Tuning:** Stratified 5-fold CV within the training set only
- **Test set:** Touched exactly once — after all model selection is complete
- **Metrics:** Accuracy, Precision, Recall (Sensitivity), Specificity, F1, ROC-AUC, Confusion Matrix
- All three models are evaluated on the **same** held-out test set

---

## Prediction Service

```python
from heart_disease.predict import predict_from_dict

result = predict_from_dict({
    "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
    "chol": 240, "fbs": 0, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
})

print(result.to_dict())
# {
#   "random_forest": {"model": "random_forest", "prediction": 0, "probability": 0.23, ...},
#   "svm":           {"model": "svm", "prediction": 0, "probability": 0.19, ...},
#   "hybrid":        {"model": "weighted_rf_svm", "prediction": 0, "probability": 0.21, ...},
#   "fusion_weight": 0.6
# }
```

**Input validation** is enforced by `PatientFeatures`: all 13 features are required with correct types and values. Invalid inputs raise a descriptive `ValueError` before any model inference occurs.

---

## Limitations

- The Cleveland dataset contains **303 instances** (1988). Results may not generalise to modern clinical populations.
- SVM probability outputs (via Plato scaling) are not guaranteed to be well-calibrated. See `reports/calibration_curves.png`.
- The hybrid model is not assumed to outperform either individual model — see the comparison table.
- This system has **not** been clinically validated and must not be used for medical decision-making.
- Missing-value imputation uses training-set statistics and may introduce bias if the test-time missing pattern differs from training.
