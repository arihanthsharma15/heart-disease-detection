"""
Model training — Random Forest and SVM pipelines.

Responsibilities
----------------
* Train a ``DummyClassifier`` baseline.
* Tune and train a ``RandomForestClassifier`` via ``GridSearchCV``.
* Tune and train an ``SVC`` (with probability estimation) via ``GridSearchCV``.
* Save each fitted pipeline to ``artifacts/`` with ``joblib``.
* Never touch the held-out test set during training or hyperparameter search.

All hyperparameter grids and constants are defined in ``config.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from heart_disease.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    RANDOM_SEED,
    RF_PARAM_GRID,
    RF_PIPELINE_PATH,
    SVM_PARAM_GRID,
    SVM_PIPELINE_PATH,
)
from heart_disease.preprocessing import build_pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_cv(n_splits: int = CV_FOLDS, seed: int = RANDOM_SEED) -> StratifiedKFold:
    """Return a ``StratifiedKFold`` with a fixed seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _grid_search(
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv: Any = None,
    verbose: int = 1,
) -> GridSearchCV:
    """Run ``GridSearchCV`` and return the fitted searcher."""
    if cv is None:
        cv = _make_cv()

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,          # refit best estimator on full X_train
        n_jobs=-1,
        verbose=verbose,
        return_train_score=False,
    )
    gs.fit(X_train, y_train)
    return gs


# ---------------------------------------------------------------------------
# Public training functions
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DummyClassifier:
    """Fit a majority-class ``DummyClassifier`` and return it.

    This establishes a lower-bound reference for all downstream models.

    Parameters
    ----------
    X_train : pd.DataFrame  Training features.
    y_train : pd.Series     Training labels.

    Returns
    -------
    DummyClassifier  Fitted dummy classifier.
    """
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)
    return dummy


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    save: bool = True,
) -> GridSearchCV:
    """Train a ``RandomForestClassifier`` pipeline via ``GridSearchCV``.

    Hyperparameter search is performed **on the training data only** using
    stratified k-fold cross-validation.

    Parameters
    ----------
    X_train : pd.DataFrame     Training features (raw, unscaled).
    y_train : pd.Series        Binary training labels.
    param_grid : dict | None   Override the default grid from ``config.py``.
    save : bool                If ``True``, save the best pipeline to ``artifacts/``.

    Returns
    -------
    GridSearchCV  Fitted searcher with ``best_estimator_`` = the best pipeline.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    rf = RandomForestClassifier(
        n_estimators=200,          # overridden by grid if included
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipeline = build_pipeline(rf, use_onehot=False)

    print("\n--- Training Random Forest ---")
    t0 = time.time()
    gs = _grid_search(pipeline, param_grid, X_train, y_train)
    elapsed = time.time() - t0

    print(f"Best params : {gs.best_params_}")
    print(f"Best CV AUC : {gs.best_score_:.4f}")
    print(f"Time elapsed: {elapsed:.1f}s")

    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, RF_PIPELINE_PATH)
        print(f"Saved → {RF_PIPELINE_PATH}")

    return gs


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    save: bool = True,
) -> GridSearchCV:
    """Train an ``SVC`` pipeline (RBF kernel, probability enabled) via ``GridSearchCV``.

    Parameters
    ----------
    X_train : pd.DataFrame     Training features (raw, unscaled).
    y_train : pd.Series        Binary training labels.
    param_grid : dict | None   Override the default grid from ``config.py``.
    save : bool                If ``True``, save the best pipeline to ``artifacts/``.

    Returns
    -------
    GridSearchCV  Fitted searcher with ``best_estimator_`` = the best pipeline.
    """
    if param_grid is None:
        param_grid = SVM_PARAM_GRID

    svm = CalibratedClassifierCV(
        estimator=SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        ensemble=False,
    )
    # SVM benefits from one-hot encoding of categorical features
    pipeline = build_pipeline(svm, use_onehot=True)

    print("\n--- Training SVM ---")
    t0 = time.time()
    gs = _grid_search(pipeline, param_grid, X_train, y_train)
    elapsed = time.time() - t0

    print(f"Best params : {gs.best_params_}")
    print(f"Best CV AUC : {gs.best_score_:.4f}")
    print(f"Time elapsed: {elapsed:.1f}s")

    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, SVM_PIPELINE_PATH)
        print(f"Saved → {SVM_PIPELINE_PATH}")

    return gs


def load_artifact(path: Path):
    """Load a saved pipeline artifact from disk.

    Parameters
    ----------
    path : Path  Path to the ``.joblib`` file.

    Returns
    -------
    Fitted sklearn Pipeline.

    Raises
    ------
    FileNotFoundError  If the artifact does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run ``python -m heart_disease.train`` first to train and save models."
        )
    return joblib.load(path)
