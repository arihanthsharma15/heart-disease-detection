"""
Unit tests for the prediction service and input validation schemas.

Tests
-----
* ``PatientFeatures`` validates correct inputs without raising.
* ``PatientFeatures`` raises ``ValueError`` for out-of-range numeric values.
* ``PatientFeatures`` raises ``ValueError`` for invalid categorical values.
* ``PatientFeatures.to_dict()`` returns keys in canonical ALL_FEATURES order.
* ``ModelPrediction.to_dict()`` serialises correctly.
* ``predict_from_dict`` raises ``KeyError`` for missing features.
* ``predict_from_dict`` raises ``ValueError`` for invalid feature values.
* End-to-end ``predict()`` returns a valid ``EnsemblePredictionResult``
  (integration test — skipped if artifacts are not present).
"""

from __future__ import annotations

import pytest
import numpy as np

from heart_disease.config import ALL_FEATURES, DECISION_THRESHOLD, DISCLAIMER
from heart_disease.schemas import (
    EnsemblePredictionResult,
    ModelPrediction,
    PatientFeatures,
)


# ---------------------------------------------------------------------------
# Helper: build a valid PatientFeatures instance
# ---------------------------------------------------------------------------

VALID_PATIENT = dict(
    age=55, sex=1, cp=2, trestbps=130, chol=240,
    fbs=0, restecg=0, thalach=150, exang=0,
    oldpeak=1.5, slope=2, ca=0, thal=3,
)


def _make_features(**overrides) -> PatientFeatures:
    kwargs = {**VALID_PATIENT, **overrides}
    return PatientFeatures(**kwargs)


# ---------------------------------------------------------------------------
# PatientFeatures — validation tests
# ---------------------------------------------------------------------------

class TestPatientFeaturesValidation:
    def test_valid_input_does_not_raise(self):
        f = _make_features()
        assert isinstance(f, PatientFeatures)

    # --- age ---
    def test_age_below_minimum_raises(self):
        with pytest.raises(ValueError, match="age"):
            _make_features(age=0)

    def test_age_above_maximum_raises(self):
        with pytest.raises(ValueError, match="age"):
            _make_features(age=200)

    # --- trestbps ---
    def test_trestbps_below_minimum_raises(self):
        with pytest.raises(ValueError, match="trestbps"):
            _make_features(trestbps=10)

    # --- chol ---
    def test_chol_above_maximum_raises(self):
        with pytest.raises(ValueError, match="chol"):
            _make_features(chol=800)

    # --- oldpeak ---
    def test_oldpeak_below_minimum_raises(self):
        with pytest.raises(ValueError, match="oldpeak"):
            _make_features(oldpeak=-10.0)

    # --- sex ---
    def test_invalid_sex_raises(self):
        with pytest.raises(ValueError, match="sex"):
            _make_features(sex=2)

    # --- cp ---
    def test_invalid_cp_raises(self):
        with pytest.raises(ValueError, match="cp"):
            _make_features(cp=5)

    # --- fbs ---
    def test_invalid_fbs_raises(self):
        with pytest.raises(ValueError, match="fbs"):
            _make_features(fbs=2)

    # --- restecg ---
    def test_invalid_restecg_raises(self):
        with pytest.raises(ValueError, match="restecg"):
            _make_features(restecg=3)

    # --- exang ---
    def test_invalid_exang_raises(self):
        with pytest.raises(ValueError, match="exang"):
            _make_features(exang=2)

    # --- slope ---
    def test_invalid_slope_raises(self):
        with pytest.raises(ValueError, match="slope"):
            _make_features(slope=4)

    # --- ca ---
    def test_invalid_ca_raises(self):
        with pytest.raises(ValueError, match="ca"):
            _make_features(ca=5)

    # --- thal ---
    def test_invalid_thal_raises(self):
        with pytest.raises(ValueError, match="thal"):
            _make_features(thal=5)

    def test_multiple_errors_all_reported(self):
        """Multiple invalid fields should all appear in the error message."""
        with pytest.raises(ValueError) as exc_info:
            _make_features(age=0, sex=9, thal=99)
        msg = str(exc_info.value)
        assert "age" in msg
        assert "sex" in msg
        assert "thal" in msg


class TestPatientFeaturesToDict:
    def test_keys_match_all_features_order(self):
        """to_dict() keys must follow the canonical ALL_FEATURES order."""
        f = _make_features()
        keys = list(f.to_dict().keys())
        assert keys == ALL_FEATURES

    def test_values_correct(self):
        f = _make_features()
        d = f.to_dict()
        assert d["age"] == 55
        assert d["thal"] == 3


# ---------------------------------------------------------------------------
# ModelPrediction serialisation
# ---------------------------------------------------------------------------

class TestModelPrediction:
    def test_to_dict_contains_required_keys(self):
        mp = ModelPrediction(
            model="random_forest",
            prediction=1,
            probability=0.75,
            threshold=DECISION_THRESHOLD,
        )
        d = mp.to_dict()
        for key in ("model", "prediction", "probability", "threshold", "disclaimer"):
            assert key in d, f"Key '{key}' missing from ModelPrediction.to_dict()."

    def test_disclaimer_is_set(self):
        mp = ModelPrediction(model="svm", prediction=0, probability=0.3, threshold=0.5)
        assert len(mp.disclaimer) > 0

    def test_null_probability_serialises(self):
        mp = ModelPrediction(model="test", prediction=0, probability=None, threshold=None)
        d = mp.to_dict()
        assert d["probability"] is None
        assert d["threshold"] is None

    def test_probability_rounded(self):
        mp = ModelPrediction(model="test", prediction=1, probability=0.123456789, threshold=0.5)
        d = mp.to_dict()
        assert d["probability"] == round(0.123456789, 4)


# ---------------------------------------------------------------------------
# predict_from_dict — validation of bad inputs
# ---------------------------------------------------------------------------

class TestPredictFromDict:
    def test_missing_key_raises_key_error(self):
        """A dict with a missing feature key should raise KeyError."""
        incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "thal"}
        from heart_disease.predict import predict_from_dict
        with pytest.raises(KeyError):
            predict_from_dict(incomplete)

    def test_invalid_value_raises_value_error(self):
        """A dict with an out-of-range value should raise ValueError."""
        bad = {**VALID_PATIENT, "age": -5}
        from heart_disease.predict import predict_from_dict
        with pytest.raises(ValueError):
            predict_from_dict(bad)


# ---------------------------------------------------------------------------
# End-to-end integration test (skipped if artifacts absent)
# ---------------------------------------------------------------------------

class TestPredictEndToEnd:
    def test_predict_returns_ensemble_result(self):
        """Integration test: predict() with valid features returns structured result."""
        from heart_disease.config import ENSEMBLE_META_PATH
        if not ENSEMBLE_META_PATH.exists():
            pytest.skip("Ensemble artifact not found — run the training pipeline first.")

        from heart_disease.predict import predict
        features = _make_features()
        result = predict(features)

        assert isinstance(result, EnsemblePredictionResult)
        assert result.random_forest.prediction in {0, 1}
        assert result.svm.prediction in {0, 1}
        assert result.hybrid.prediction in {0, 1}
        assert 0.0 <= result.fusion_weight <= 1.0

        for model_pred in (result.random_forest, result.svm, result.hybrid):
            if model_pred.probability is not None:
                assert 0.0 <= model_pred.probability <= 1.0
