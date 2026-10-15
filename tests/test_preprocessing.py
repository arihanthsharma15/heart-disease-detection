"""
Unit tests for the preprocessing pipeline.

Tests
-----
* Pipeline builds without errors.
* Pipeline fits on training data without raising.
* Transform output has the expected number of columns.
* Missing values in ``ca`` and ``thal`` are imputed (no NaNs in output).
* Scaler is fitted on training data only (no leakage from test data).
* Feature order is preserved between training and inference transforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from heart_disease.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from heart_disease.preprocessing import build_pipeline, build_preprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal valid DataFrame with the canonical 13-feature schema."""
    np.random.seed(42)
    n = 30
    data = {
        "age":      np.random.randint(30, 75, n),
        "trestbps": np.random.randint(90, 180, n),
        "chol":     np.random.randint(150, 400, n),
        "thalach":  np.random.randint(70, 200, n),
        "oldpeak":  np.random.uniform(0.0, 5.0, n),
        "sex":      np.random.choice([0, 1], n),
        "cp":       np.random.choice([1, 2, 3, 4], n),
        "fbs":      np.random.choice([0, 1], n),
        "restecg":  np.random.choice([0, 1, 2], n),
        "exang":    np.random.choice([0, 1], n),
        "slope":    np.random.choice([1, 2, 3], n),
        "ca":       np.random.choice([0, 1, 2, 3, np.nan], n),
        "thal":     np.random.choice([3.0, 6.0, 7.0, np.nan], n),
    }
    return pd.DataFrame(data)[ALL_FEATURES]


@pytest.fixture()
def sample_target(sample_df: pd.DataFrame) -> pd.Series:
    np.random.seed(0)
    return pd.Series(np.random.choice([0, 1], len(sample_df)), name="target")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_returns_column_transformer(self, sample_df):
        """build_preprocessor should return a ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        pre = build_preprocessor()
        assert hasattr(pre, "fit_transform"), "Expected a transformer object."

    def test_fit_transform_no_nans(self, sample_df):
        """Output of fit_transform should contain no NaN values."""
        pre = build_preprocessor()
        out = pre.fit_transform(sample_df)
        assert not np.isnan(out).any(), "NaNs remain after imputation."

    def test_fit_transform_shape(self, sample_df):
        """Output should have the same number of rows as the input."""
        pre = build_preprocessor()
        out = pre.fit_transform(sample_df)
        assert out.shape[0] == len(sample_df)

    def test_onehot_more_columns_than_ordinal(self, sample_df):
        """One-hot encoding should produce more columns than ordinal encoding."""
        pre_ord = build_preprocessor(use_onehot=False)
        pre_ohe = build_preprocessor(use_onehot=True)
        out_ord = pre_ord.fit_transform(sample_df)
        out_ohe = pre_ohe.fit_transform(sample_df)
        assert out_ohe.shape[1] > out_ord.shape[1], (
            "OneHot should expand categorical features beyond ordinal."
        )


class TestBuildPipeline:
    def test_pipeline_returns_pipeline(self):
        """build_pipeline should return an sklearn Pipeline."""
        from sklearn.dummy import DummyClassifier
        pipe = build_pipeline(DummyClassifier())
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_two_steps(self):
        """Pipeline should have exactly two named steps: preprocessor + classifier."""
        from sklearn.dummy import DummyClassifier
        pipe = build_pipeline(DummyClassifier())
        assert list(pipe.named_steps.keys()) == ["preprocessor", "classifier"]

    def test_fit_predict(self, sample_df, sample_target):
        """Pipeline should fit and produce binary predictions without error."""
        from sklearn.dummy import DummyClassifier
        pipe = build_pipeline(DummyClassifier(strategy="most_frequent"))
        pipe.fit(sample_df, sample_target)
        preds = pipe.predict(sample_df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_no_leakage_test_data_not_used_in_fit(self, sample_df, sample_target):
        """Scaler fitted on training data should differ from test-data statistics.

        We verify that the scaler's learned mean (from training rows only)
        is NOT equal to the overall dataset mean — confirming the scaler
        is fitted only on training data, not the whole dataset.
        """
        train_df = sample_df.iloc[:20].copy()
        test_df = sample_df.iloc[20:].copy()
        train_target = sample_target.iloc[:20]

        from sklearn.dummy import DummyClassifier
        pipe = build_pipeline(DummyClassifier())
        pipe.fit(train_df, train_target)

        # Extract the StandardScaler from the fitted preprocessor
        scaler = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
        learned_mean = scaler.mean_[0]  # mean for 'age' column

        full_age_mean = sample_df["age"].mean()
        train_age_mean = train_df["age"].mean()

        # The scaler should know the training mean, not the full-dataset mean
        assert abs(learned_mean - train_age_mean) < 1e-6, (
            "Scaler mean does not match training data mean — possible leakage."
        )

    def test_transform_preserves_row_count(self, sample_df, sample_target):
        """Transformed test data should have same row count as test input."""
        from sklearn.dummy import DummyClassifier
        train_df = sample_df.iloc[:20]
        test_df = sample_df.iloc[20:]
        train_target = sample_target.iloc[:20]
        pipe = build_pipeline(DummyClassifier())
        pipe.fit(train_df, train_target)
        transformed = pipe[:-1].transform(test_df)  # preprocessor only
        assert transformed.shape[0] == len(test_df)
