"""
Weighted probability fusion ensemble.

Formula
-------
    P_hybrid = w * P_RF + (1 - w) * P_SVM

where 0 ≤ w ≤ 1.

The fusion weight ``w`` is selected by maximising ROC-AUC on a
stratified validation fold of the **training data only** — the test set
is never touched during weight selection.

Responsibilities
----------------
* ``select_fusion_weight`` — grid-search w on training data via cross-validation.
* ``EnsemblePredictor``    — wraps two fitted pipelines and applies fusion at
                             inference time.
* Persist / load ensemble metadata (w, threshold) alongside model pipelines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from heart_disease.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    DECISION_THRESHOLD,
    ENSEMBLE_META_PATH,
    FUSION_WEIGHT_CANDIDATES,
    POSITIVE_CLASS_LABEL,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Weight selection
# ---------------------------------------------------------------------------

def select_fusion_weight(
    rf_pipeline,
    svm_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weight_candidates: list[float] | None = None,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> tuple[float, float]:
    """Select the optimal fusion weight via cross-validated ROC-AUC.

    Both pipelines are re-fitted on each training fold and evaluated on
    the corresponding validation fold. Weights are evaluated using the
    *average validation AUC* across all folds, so the held-out test set
    is never involved in weight selection.

    Parameters
    ----------
    rf_pipeline : fitted sklearn Pipeline  Random Forest pipeline.
    svm_pipeline : fitted sklearn Pipeline SVM pipeline.
    X_train : pd.DataFrame                 Training features.
    y_train : pd.Series                    Binary training labels.
    weight_candidates : list[float] | None Candidate w values (0 … 1).
                                           Defaults to ``config.FUSION_WEIGHT_CANDIDATES``.
    n_splits : int   Number of CV folds (default from ``config``).
    random_state : int  Seed for reproducibility.
    verbose : bool   Print progress.

    Returns
    -------
    best_w : float      The weight that maximised mean CV AUC.
    best_auc : float    The corresponding mean CV AUC.
    """
    if weight_candidates is None:
        weight_candidates = FUSION_WEIGHT_CANDIDATES

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    weight_aucs: dict[float, list[float]] = {w: [] for w in weight_candidates}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Clone and refit both pipelines on this fold's training data
        from sklearn.base import clone
        rf_fold = clone(rf_pipeline).fit(X_fold_train, y_fold_train)
        svm_fold = clone(svm_pipeline).fit(X_fold_train, y_fold_train)

        p_rf = rf_fold.predict_proba(X_val)[:, POSITIVE_CLASS_LABEL]
        p_svm = svm_fold.predict_proba(X_val)[:, POSITIVE_CLASS_LABEL]

        for w in weight_candidates:
            p_hybrid = w * p_rf + (1.0 - w) * p_svm
            auc = roc_auc_score(y_val, p_hybrid)
            weight_aucs[w].append(auc)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_splits} complete")

    mean_aucs = {w: float(np.mean(aucs)) for w, aucs in weight_aucs.items()}
    best_w = max(mean_aucs, key=mean_aucs.__getitem__)
    best_auc = mean_aucs[best_w]

    if verbose:
        print(f"\nFusion weight selected: w = {best_w:.2f}  "
              f"(mean CV AUC = {best_auc:.4f})")

    return best_w, best_auc


# ---------------------------------------------------------------------------
# Ensemble predictor
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePredictor:
    """Weighted probability fusion of a Random Forest and an SVM pipeline.

    Attributes
    ----------
    rf_pipeline : fitted sklearn Pipeline
    svm_pipeline : fitted sklearn Pipeline
    fusion_weight : float
        ``w`` in ``P_hybrid = w * P_RF + (1-w) * P_SVM``.
        Selected on validation data; never tuned against the test set.
    threshold : float
        Decision threshold applied to ``P_hybrid`` to produce a binary prediction.
    cv_auc : float | None
        Mean cross-validated AUC achieved at the selected ``fusion_weight``.
        Stored for reporting; ``None`` if not available.
    """

    rf_pipeline: object
    svm_pipeline: object
    fusion_weight: float
    threshold: float = DECISION_THRESHOLD
    cv_auc: Optional[float] = None

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------

    def predict_proba_rf(self, X: pd.DataFrame) -> np.ndarray:
        """Return RF probability of the positive class."""
        return self.rf_pipeline.predict_proba(X)[:, POSITIVE_CLASS_LABEL]

    def predict_proba_svm(self, X: pd.DataFrame) -> np.ndarray:
        """Return SVM probability of the positive class."""
        return self.svm_pipeline.predict_proba(X)[:, POSITIVE_CLASS_LABEL]

    def predict_proba_hybrid(self, X: pd.DataFrame) -> np.ndarray:
        """Return fused probability: w * P_RF + (1-w) * P_SVM."""
        p_rf = self.predict_proba_rf(X)
        p_svm = self.predict_proba_svm(X)
        return self.fusion_weight * p_rf + (1.0 - self.fusion_weight) * p_svm

    def predict_rf(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard predictions from the Random Forest pipeline."""
        return self.rf_pipeline.predict(X)

    def predict_svm(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard predictions from the SVM pipeline."""
        return self.svm_pipeline.predict(X)

    def predict_hybrid(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard predictions from the hybrid model using ``self.threshold``."""
        proba = self.predict_proba_hybrid(X)
        return (proba >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None) -> None:
        """Serialise this predictor to disk.

        Parameters
        ----------
        path : Path | None  Target path.  Defaults to ``config.ENSEMBLE_META_PATH``.
        """
        if path is None:
            path = ENSEMBLE_META_PATH
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Ensemble predictor saved → {path}")

    @classmethod
    def load(cls, path=None) -> "EnsemblePredictor":
        """Load a persisted ``EnsemblePredictor`` from disk.

        Parameters
        ----------
        path : Path | None  Source path.  Defaults to ``config.ENSEMBLE_META_PATH``.

        Raises
        ------
        FileNotFoundError  If the artifact does not exist.
        """
        if path is None:
            path = ENSEMBLE_META_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Ensemble artifact not found: {path}\n"
                "Run the training script first."
            )
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not an EnsemblePredictor: {type(obj)}")
        return obj
