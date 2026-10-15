"""
Unit tests for the weighted probability fusion ensemble.

Tests
-----
* EnsemblePredictor produces probabilities in [0, 1].
* Fusion formula: w=1 → pure RF, w=0 → pure SVM.
* Hard predictions match threshold applied to hybrid probabilities.
* select_fusion_weight returns a w in [0, 1].
* Predictor can be serialised and deserialised (round-trip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from heart_disease.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DECISION_THRESHOLD,
    FUSION_WEIGHT_CANDIDATES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
)
from heart_disease.ensemble import EnsemblePredictor, select_fusion_weight
from heart_disease.preprocessing import build_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(n: int = 60, seed: int = 42):
    """Create synthetic data that mimics the Cleveland feature schema."""
    rng = np.random.default_rng(seed)
    data = {
        "age":      rng.integers(30, 75, n),
        "trestbps": rng.integers(90, 180, n),
        "chol":     rng.integers(150, 400, n),
        "thalach":  rng.integers(70, 200, n),
        "oldpeak":  rng.uniform(0.0, 5.0, n),
        "sex":      rng.choice([0, 1], n),
        "cp":       rng.choice([1, 2, 3, 4], n),
        "fbs":      rng.choice([0, 1], n),
        "restecg":  rng.choice([0, 1, 2], n),
        "exang":    rng.choice([0, 1], n),
        "slope":    rng.choice([1, 2, 3], n),
        "ca":       rng.choice([0, 1, 2, 3], n).astype(float),
        "thal":     rng.choice([3.0, 6.0, 7.0], n),
    }
    X = pd.DataFrame(data)[ALL_FEATURES]
    y = pd.Series(rng.choice([0, 1], n), name="target")
    return X, y


@pytest.fixture(scope="module")
def fitted_rf_pipeline():
    """Fit a Random Forest pipeline on synthetic training data."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = _make_synthetic_data(n=60)
    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED))
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def fitted_svm_pipeline():
    """Fit an SVM pipeline on synthetic training data."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC
    X, y = _make_synthetic_data(n=60)
    pipe = build_pipeline(
        CalibratedClassifierCV(estimator=SVC(random_state=RANDOM_SEED), ensemble=False),
        use_onehot=True,
    )
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="module")
def ensemble_predictor(fitted_rf_pipeline, fitted_svm_pipeline):
    return EnsemblePredictor(
        rf_pipeline=fitted_rf_pipeline,
        svm_pipeline=fitted_svm_pipeline,
        fusion_weight=0.5,
        threshold=DECISION_THRESHOLD,
    )


@pytest.fixture(scope="module")
def X_test():
    X, _ = _make_synthetic_data(n=20, seed=99)
    return X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsemblePredictor:
    def test_proba_rf_in_unit_interval(self, ensemble_predictor, X_test):
        p = ensemble_predictor.predict_proba_rf(X_test)
        assert np.all((p >= 0) & (p <= 1)), "RF probabilities outside [0, 1]."

    def test_proba_svm_in_unit_interval(self, ensemble_predictor, X_test):
        p = ensemble_predictor.predict_proba_svm(X_test)
        assert np.all((p >= 0) & (p <= 1)), "SVM probabilities outside [0, 1]."

    def test_proba_hybrid_in_unit_interval(self, ensemble_predictor, X_test):
        p = ensemble_predictor.predict_proba_hybrid(X_test)
        assert np.all((p >= 0) & (p <= 1)), "Hybrid probabilities outside [0, 1]."

    def test_fusion_w1_equals_rf(self, fitted_rf_pipeline, fitted_svm_pipeline, X_test):
        """When w=1, hybrid == RF."""
        ep = EnsemblePredictor(
            rf_pipeline=fitted_rf_pipeline,
            svm_pipeline=fitted_svm_pipeline,
            fusion_weight=1.0,
        )
        p_hybrid = ep.predict_proba_hybrid(X_test)
        p_rf = ep.predict_proba_rf(X_test)
        np.testing.assert_allclose(p_hybrid, p_rf, atol=1e-8)

    def test_fusion_w0_equals_svm(self, fitted_rf_pipeline, fitted_svm_pipeline, X_test):
        """When w=0, hybrid == SVM."""
        ep = EnsemblePredictor(
            rf_pipeline=fitted_rf_pipeline,
            svm_pipeline=fitted_svm_pipeline,
            fusion_weight=0.0,
        )
        p_hybrid = ep.predict_proba_hybrid(X_test)
        p_svm = ep.predict_proba_svm(X_test)
        np.testing.assert_allclose(p_hybrid, p_svm, atol=1e-8)

    def test_hard_predictions_match_threshold(self, ensemble_predictor, X_test):
        """Hard predictions must be consistent with the probability + threshold."""
        p_hybrid = ensemble_predictor.predict_proba_hybrid(X_test)
        expected = (p_hybrid >= ensemble_predictor.threshold).astype(int)
        actual = ensemble_predictor.predict_hybrid(X_test)
        np.testing.assert_array_equal(actual, expected)

    def test_predictions_binary(self, ensemble_predictor, X_test):
        """All hard predictions should be 0 or 1."""
        for pred_fn in [
            ensemble_predictor.predict_rf,
            ensemble_predictor.predict_svm,
            ensemble_predictor.predict_hybrid,
        ]:
            preds = pred_fn(X_test)
            assert set(np.unique(preds)).issubset({0, 1}), (
                f"{pred_fn.__name__} produced non-binary predictions."
            )


class TestSelectFusionWeight:
    def test_returns_weight_in_unit_interval(self, fitted_rf_pipeline, fitted_svm_pipeline):
        X, y = _make_synthetic_data(n=60)
        # Use a small subset of candidate weights for speed
        best_w, best_auc = select_fusion_weight(
            fitted_rf_pipeline, fitted_svm_pipeline, X, y,
            weight_candidates=[0.0, 0.5, 1.0],
            n_splits=2,
            verbose=False,
        )
        assert 0.0 <= best_w <= 1.0, f"best_w={best_w} is outside [0, 1]."

    def test_returns_auc_in_unit_interval(self, fitted_rf_pipeline, fitted_svm_pipeline):
        X, y = _make_synthetic_data(n=60)
        _, best_auc = select_fusion_weight(
            fitted_rf_pipeline, fitted_svm_pipeline, X, y,
            weight_candidates=[0.0, 0.5, 1.0],
            n_splits=2,
            verbose=False,
        )
        assert 0.0 <= best_auc <= 1.0, f"best_auc={best_auc} is outside [0, 1]."


class TestEnsemblePersistence:
    def test_save_and_load_roundtrip(self, ensemble_predictor, X_test, tmp_path):
        """Save and reload the ensemble; predictions should be identical."""
        save_path = tmp_path / "ensemble_test.joblib"
        ensemble_predictor.save(path=save_path)

        loaded = EnsemblePredictor.load(path=save_path)
        p_original = ensemble_predictor.predict_proba_hybrid(X_test)
        p_loaded   = loaded.predict_proba_hybrid(X_test)
        np.testing.assert_allclose(p_original, p_loaded, atol=1e-8)

    def test_load_missing_file_raises(self, tmp_path):
        """Loading a non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            EnsemblePredictor.load(path=tmp_path / "does_not_exist.joblib")
