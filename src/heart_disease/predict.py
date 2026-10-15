"""
Prediction service — inference entry point.

Accepts validated ``PatientFeatures``, runs all three models, and returns
a structured ``EnsemblePredictionResult``.

Responsibilities
----------------
* Load trained artifact pipelines from disk (lazy, cached).
* Accept a ``PatientFeatures`` instance (already validated by the schema).
* Convert features to a one-row DataFrame in the canonical column order.
* Obtain individual RF and SVM predictions and probabilities.
* Compute hybrid fused probability and hard prediction.
* Return a fully populated ``EnsemblePredictionResult``.

This module does **not** train models or access the test set.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path

from heart_disease.config import (
    ALL_FEATURES,
    DECISION_THRESHOLD,
    DISCLAIMER,
    ENSEMBLE_META_PATH,
    POSITIVE_CLASS_LABEL,
    RF_PIPELINE_PATH,
    SVM_PIPELINE_PATH,
)
from heart_disease.ensemble import EnsemblePredictor
from heart_disease.schemas import (
    EnsemblePredictionResult,
    ModelPrediction,
    PatientFeatures,
)
from heart_disease.train import load_artifact


# ---------------------------------------------------------------------------
# Lazy artifact loading (loaded once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ensemble() -> EnsemblePredictor:
    """Load the saved ``EnsemblePredictor`` from disk (cached after first call).

    Raises
    ------
    FileNotFoundError
        If the ensemble artifact does not exist.
    """
    return EnsemblePredictor.load(ENSEMBLE_META_PATH)


# ---------------------------------------------------------------------------
# Feature conversion helper
# ---------------------------------------------------------------------------

def _features_to_dataframe(features: PatientFeatures) -> pd.DataFrame:
    """Convert a ``PatientFeatures`` instance to a single-row ``DataFrame``."""
    row = features.to_dict()
    return pd.DataFrame([row], columns=ALL_FEATURES)


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict(features: PatientFeatures) -> EnsemblePredictionResult:
    """Run all three models and return a structured prediction result.

    Parameters
    ----------
    features : PatientFeatures
        Validated patient features.  The ``PatientFeatures`` dataclass
        validates ranges and categorical values during construction; any
        out-of-range or unknown values raise ``ValueError`` before this
        function is called.

    Returns
    -------
    EnsemblePredictionResult
        Contains individual RF and SVM predictions plus the hybrid fused
        prediction.

    Raises
    ------
    FileNotFoundError
        If model artifacts have not been generated (run training first).
    TypeError
        If ``features`` is not a ``PatientFeatures`` instance.
    """
    if not isinstance(features, PatientFeatures):
        raise TypeError(
            f"Expected PatientFeatures, got {type(features).__name__}. "
            "Construct a PatientFeatures object with all 13 required fields."
        )

    ensemble: EnsemblePredictor = _load_ensemble()
    X = _features_to_dataframe(features)

    # --- Random Forest ---
    p_rf = float(ensemble.predict_proba_rf(X)[0])
    pred_rf = int(p_rf >= DECISION_THRESHOLD)
    rf_result = ModelPrediction(
        model="random_forest",
        prediction=pred_rf,
        probability=p_rf,
        threshold=DECISION_THRESHOLD,
    )

    # --- SVM ---
    p_svm = float(ensemble.predict_proba_svm(X)[0])
    pred_svm = int(p_svm >= DECISION_THRESHOLD)
    svm_result = ModelPrediction(
        model="svm",
        prediction=pred_svm,
        probability=p_svm,
        threshold=DECISION_THRESHOLD,
    )

    # --- Hybrid ---
    p_hybrid = float(ensemble.predict_proba_hybrid(X)[0])
    pred_hybrid = int(p_hybrid >= ensemble.threshold)
    hybrid_result = ModelPrediction(
        model="weighted_rf_svm",
        prediction=pred_hybrid,
        probability=p_hybrid,
        threshold=ensemble.threshold,
    )

    return EnsemblePredictionResult(
        random_forest=rf_result,
        svm=svm_result,
        hybrid=hybrid_result,
        fusion_weight=ensemble.fusion_weight,
    )


def predict_from_dict(raw: dict) -> EnsemblePredictionResult:
    """Convenience wrapper: construct ``PatientFeatures`` from a plain dict and predict.

    Parameters
    ----------
    raw : dict
        Must contain all 13 feature keys with appropriate values.

    Returns
    -------
    EnsemblePredictionResult

    Raises
    ------
    ValueError
        If any feature value is missing, out of range, or invalid.
    KeyError
        If a required feature key is absent from ``raw``.
    """
    try:
        features = PatientFeatures(
            age=raw["age"],
            sex=raw["sex"],
            cp=raw["cp"],
            trestbps=raw["trestbps"],
            chol=raw["chol"],
            fbs=raw["fbs"],
            restecg=raw["restecg"],
            thalach=raw["thalach"],
            exang=raw["exang"],
            oldpeak=float(raw["oldpeak"]),
            slope=raw["slope"],
            ca=raw["ca"],
            thal=raw["thal"],
        )
    except KeyError as exc:
        raise KeyError(
            f"Missing required feature: {exc}. "
            f"All 13 features are required: {ALL_FEATURES}"
        ) from exc

    return predict(features)
