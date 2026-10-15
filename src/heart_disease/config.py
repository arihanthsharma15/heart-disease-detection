"""
Central configuration: paths, random seed, feature lists, and hyperparameter grids.

All other modules import constants from here — changing a value once propagates
everywhere. Do not hardcode paths or seeds in individual modules.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[2]  # repo root
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
REPORTS_DIR: Path = ROOT_DIR / "reports"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Saved artifact file names
RF_PIPELINE_PATH: Path = ARTIFACTS_DIR / "rf_pipeline.joblib"
SVM_PIPELINE_PATH: Path = ARTIFACTS_DIR / "svm_pipeline.joblib"
ENSEMBLE_META_PATH: Path = ARTIFACTS_DIR / "ensemble_meta.joblib"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
UCI_DATASET_ID: int = 45          # Heart Disease dataset

# Raw target column name as returned by ucimlrepo
TARGET_COL: str = "num"

# Binary mapping: 0 → no disease; 1,2,3,4 → disease present
DISEASE_THRESHOLD: int = 1        # raw values >= threshold become label 1
POSITIVE_CLASS_LABEL: int = 1     # label that means "disease present"

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Valid value ranges for inference-time validation
# ---------------------------------------------------------------------------
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age":      (1.0,   120.0),
    "trestbps": (50.0,  300.0),
    "chol":     (50.0,  700.0),
    "thalach":  (40.0,  280.0),
    "oldpeak":  (-5.0,  10.0),
}

FEATURE_VALID_VALUES: dict[str, set[int]] = {
    "sex":     {0, 1},
    "cp":      {1, 2, 3, 4},
    "fbs":     {0, 1},
    "restecg": {0, 1, 2},
    "exang":   {0, 1},
    "slope":   {1, 2, 3},
    "ca":      {0, 1, 2, 3},
    "thal":    {3, 6, 7},
}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Hyperparameter grids (searched on training data only)
# ---------------------------------------------------------------------------
RF_PARAM_GRID: dict = {
    "classifier__max_depth":         [None, 5, 10, 20],
    "classifier__min_samples_leaf":  [1, 2, 4],
    "classifier__n_estimators":      [100, 200],
}

SVM_PARAM_GRID: dict = {
    "classifier__estimator__C":     [0.1, 1.0, 10.0, 100.0],
    "classifier__estimator__gamma": ["scale", "auto"],
}

# ---------------------------------------------------------------------------
# Ensemble fusion
# ---------------------------------------------------------------------------
# Weight w in [0, 1]: P_hybrid = w * P_RF + (1-w) * P_SVM
FUSION_WEIGHT_CANDIDATES: list[float] = [round(x * 0.05, 2) for x in range(21)]  # 0.00 … 1.00
DECISION_THRESHOLD: float = 0.50

# ---------------------------------------------------------------------------
# Clinical Risk Stratification Tiers
# ---------------------------------------------------------------------------
# Low Risk:      probability < 0.55 (0% to 54.9%)
# Moderate Risk: 0.55 <= probability <= 0.75 (55% to 75%)
# High Risk:     probability > 0.75 (75.1% to 100%)
RISK_TIER_MODERATE_MIN: float = 0.55
RISK_TIER_HIGH_MIN: float = 0.75

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER: str = (
    "Research use only; not a medical diagnosis. "
    "This system has not been clinically validated. "
    "Do not use for clinical decision-making."
)

