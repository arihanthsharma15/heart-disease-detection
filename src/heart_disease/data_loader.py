"""
Data loading and preparation for the UCI Heart Disease (Cleveland) dataset.

Responsibilities
----------------
* Fetch the dataset from UCI ML Repository via ``ucimlrepo``.
* Validate the feature schema against the expected column list.
* Apply the documented binary target mapping (0 = no disease, 1 = disease).
* Perform a stratified train/test split — no preprocessing leakage.
* Provide an EDA summary (printed to stdout or returned as a dict).

This module does **not** preprocess features (scaling, encoding, imputation).
That is handled exclusively in ``preprocessing.py`` to keep the pipeline clean.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heart_disease.config import (
    ALL_FEATURES,
    DISEASE_THRESHOLD,
    POSITIVE_CLASS_LABEL,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
    UCI_DATASET_ID,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_target(series: pd.Series) -> pd.Series:
    """Map raw ``num`` (0–4) to binary labels.

    Mapping
    -------
    0        → 0  (no disease — < 50 % diameter narrowing)
    1, 2, 3, 4 → 1  (disease present — ≥ 50 % narrowing in ≥ 1 major vessel)

    This follows the convention in Detrano et al. (1989) and is the protocol
    used in every published Cleveland-subset experiment.
    """
    return (series >= DISEASE_THRESHOLD).astype(int)


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if required columns are missing from *df*."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Loaded dataset is missing expected feature columns: {missing}.\n"
            "This may indicate a dataset version mismatch. "
            "Only the UCI Cleveland subset (id=45) is supported."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cleveland_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the Cleveland Heart Disease dataset and apply the binary target mapping.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with exactly the 13 columns listed in ``config.ALL_FEATURES``,
        in that order. Missing values (``ca``, ``thal``) are preserved as ``NaN``;
        imputation happens inside the preprocessing pipeline.
    y : pd.Series
        Binary target — 0 (no disease) or 1 (disease present).

    Raises
    ------
    ImportError
        If ``ucimlrepo`` is not installed.
    ValueError
        If the fetched dataset does not contain the expected feature columns.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "The 'ucimlrepo' package is required to download the dataset.\n"
            "Install it with: pip install ucimlrepo"
        ) from exc

    print(f"Fetching UCI Heart Disease dataset (id={UCI_DATASET_ID}) …")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    # ucimlrepo returns X and y separately
    X_raw: pd.DataFrame = dataset.data.features.copy()
    y_raw: pd.Series = dataset.data.targets.iloc[:, 0].copy()  # 'num' column

    # Validate schema
    _validate_schema(X_raw)

    # Enforce canonical column order and select only the 13 expected features
    X = X_raw[ALL_FEATURES].copy()

    # Binary target mapping
    y = _binary_target(y_raw)
    y.name = "target"

    # Sanity checks
    assert len(X) == len(y), "Feature and target row counts do not match."
    assert set(y.unique()).issubset({0, 1}), "Binary target contains unexpected values."

    print(
        f"Dataset loaded: {len(X)} rows × {len(X.columns)} features | "
        f"Positive class ({POSITIVE_CLASS_LABEL}): {y.sum()} "
        f"({y.mean() * 100:.1f} %)"
    )
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    X : pd.DataFrame   Feature matrix.
    y : pd.Series      Binary target.
    test_size : float  Fraction of data to hold out as test set (default 0.20).
    random_state : int Reproducibility seed (default ``RANDOM_SEED``).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(
        f"Split: {len(X_train)} train / {len(X_test)} test  "
        f"(train positive rate: {y_train.mean() * 100:.1f} %  "
        f"test positive rate: {y_test.mean() * 100:.1f} %)"
    )
    return X_train, X_test, y_train, y_test


def eda_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    """Return a structured EDA summary and print it to stdout.

    Parameters
    ----------
    X : pd.DataFrame  Feature matrix.
    y : pd.Series     Binary target.

    Returns
    -------
    dict containing shape, dtypes, missing value counts, class balance,
    and descriptive statistics.
    """
    n_rows, n_cols = X.shape
    missing_counts = X.isnull().sum()
    missing_pct = (missing_counts / n_rows * 100).round(2)
    class_counts = y.value_counts().sort_index()
    class_pct = (y.value_counts(normalize=True).sort_index() * 100).round(2)

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": X.dtypes.astype(str).to_dict(),
        "missing_counts": missing_counts[missing_counts > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "class_counts": class_counts.to_dict(),
        "class_pct": class_pct.to_dict(),
        "duplicates": int(X.duplicated().sum()),
        "describe": X.describe().to_dict(),
    }

    print("\n" + "=" * 60)
    print("EDA SUMMARY — UCI Heart Disease (Cleveland)")
    print("=" * 60)
    print(f"Shape : {n_rows} rows × {n_cols} columns")
    print(f"Duplicates : {summary['duplicates']}")
    print("\nMissing values:")
    if summary["missing_counts"]:
        for col, cnt in summary["missing_counts"].items():
            print(f"  {col:<12} {cnt:>4} ({summary['missing_pct'][col]:.1f} %)")
    else:
        print("  None")
    print("\nTarget class balance:")
    for label, cnt in class_counts.items():
        print(f"  label={label}  {cnt:>4} ({class_pct[label]:.1f} %)")
    print("=" * 60 + "\n")

    return summary
