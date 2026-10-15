"""
Model evaluation — metrics, comparison table, confusion matrices, ROC curves.

This module is called **once** with the held-out test set, only after all
model selection and hyperparameter tuning is complete.

Responsibilities
----------------
* Compute accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix.
* Build a side-by-side comparison table for RF, SVM, and hybrid models.
* Generate and save confusion matrix plots and a ROC-curve overlay.
* Print a full classification report.

Nothing in this module modifies, retrains, or selects among models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from heart_disease.config import DECISION_THRESHOLD, POSITIVE_CLASS_LABEL, REPORTS_DIR


# ---------------------------------------------------------------------------
# Single-model metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float | None]:
    """Compute classification metrics for a single model.

    Parameters
    ----------
    y_true : array-like  Ground-truth binary labels.
    y_pred : array-like  Hard binary predictions.
    y_prob : array-like | None  Predicted probabilities for the positive class.
                                Required for ROC-AUC; pass ``None`` if unavailable.
    threshold : float    Decision threshold (recorded for documentation).

    Returns
    -------
    dict with keys: accuracy, precision, recall, specificity, f1, roc_auc, threshold.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    roc_auc = None
    if y_prob is not None:
        roc_auc = float(roc_auc_score(y_true, np.asarray(y_prob)))

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":     roc_auc,
        "threshold":   threshold,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def comparison_table(metrics_dict: dict[str, dict]) -> pd.DataFrame:
    """Build and print a model comparison table.

    Parameters
    ----------
    metrics_dict : dict[str, dict]
        Keys are model names (e.g. ``"Random Forest"``);
        values are dicts returned by ``compute_metrics``.

    Returns
    -------
    pd.DataFrame  Comparison table (models as rows, metrics as columns).
    """
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model":       model_name,
            "Accuracy":    f"{m['accuracy']:.4f}",
            "Precision":   f"{m['precision']:.4f}",
            "Recall":      f"{m['recall']:.4f}",
            "Specificity": f"{m['specificity']:.4f}" if m["specificity"] is not None else "—",
            "F1":          f"{m['f1']:.4f}",
            "ROC-AUC":     f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "—",
            "Threshold":   f"{m['threshold']:.2f}",
        })
    df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Test Set Results")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")

    return df


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def print_classification_reports(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    """Print scikit-learn classification reports for each model.

    Parameters
    ----------
    y_true : array-like     Ground-truth labels.
    predictions : dict      Model name → hard predictions.
    """
    for model_name, y_pred in predictions.items():
        print(f"\n--- Classification Report: {model_name} ---")
        print(classification_report(
            y_true, y_pred,
            target_names=["No Disease (0)", "Disease (1)"],
        ))


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    save_dir: Path = REPORTS_DIR,
    show: bool = False,
) -> None:
    """Plot and save confusion matrices for each model.

    Parameters
    ----------
    y_true : array-like     Ground-truth labels.
    predictions : dict      Model name → hard predictions.
    save_dir : Path         Directory to save ``confusion_matrix_<model>.png`` files.
    show : bool             If ``True``, call ``plt.show()`` (non-headless environments).
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Disease", "Disease"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(model_name, fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_dir / "confusion_matrices.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Confusion matrices saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    save_dir: Path = REPORTS_DIR,
    show: bool = False,
) -> None:
    """Plot and save an overlaid ROC curve for each model.

    Parameters
    ----------
    y_true : array-like      Ground-truth binary labels.
    probabilities : dict     Model name → predicted positive-class probabilities.
    save_dir : Path          Directory to save ``roc_curves.png``.
    show : bool              If ``True``, call ``plt.show()``.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "green", "red"]

    for (model_name, y_prob), color in zip(probabilities.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{model_name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = save_dir / "roc_curves.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"ROC curves saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    n_bins: int = 10,
    save_dir: Path = REPORTS_DIR,
    show: bool = False,
) -> None:
    """Plot and save calibration (reliability) diagrams.

    Parameters
    ----------
    y_true : array-like      Ground-truth binary labels.
    probabilities : dict     Model name → predicted positive-class probabilities.
    n_bins : int             Number of bins for calibration curve.
    save_dir : Path          Directory to save ``calibration_curves.png``.
    show : bool              If ``True``, call ``plt.show()``.
    """
    from sklearn.calibration import calibration_curve

    save_dir.mkdir(parents=True, exist_ok=True)
    colors = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    for (model_name, y_prob), color in zip(probabilities.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", color=color, lw=2, label=model_name)

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = save_dir / "calibration_curves.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Calibration curves saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)
