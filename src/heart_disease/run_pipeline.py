"""
End-to-end training + evaluation script.

Run from the repository root:

    python -m heart_disease.run_pipeline

Phases executed
---------------
1. Load the UCI Heart Disease (Cleveland) dataset.
2. Print an EDA summary.
3. Stratified 80/20 train/test split.
4. Train baseline DummyClassifier.
5. Train Random Forest with GridSearchCV.
6. Train SVM with GridSearchCV.
7. Select fusion weight on training data via cross-validation.
8. Build and save EnsemblePredictor.
9. Evaluate all three models on the held-out test set.
10. Save comparison table, confusion matrices, ROC curves, and calibration plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from heart_disease.config import (
    ARTIFACTS_DIR,
    DECISION_THRESHOLD,
    RANDOM_SEED,
    REPORTS_DIR,
)
from heart_disease.data_loader import eda_summary, load_cleveland_dataset, split_data
from heart_disease.ensemble import EnsemblePredictor, select_fusion_weight
from heart_disease.evaluate import (
    comparison_table,
    compute_metrics,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_roc_curves,
    print_classification_reports,
)
from heart_disease.train import train_baseline, train_random_forest, train_svm
from heart_disease.config import RF_PIPELINE_PATH, SVM_PIPELINE_PATH, ENSEMBLE_META_PATH


def run() -> None:
    """Execute the full pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("  HEART DISEASE DETECTION — ML PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase B: Load data + EDA
    # ------------------------------------------------------------------
    X, y = load_cleveland_dataset()
    eda_summary(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # Phase C: Train models
    # ------------------------------------------------------------------
    dummy = train_baseline(X_train, y_train)
    rf_gs = train_random_forest(X_train, y_train, save=True)
    svm_gs = train_svm(X_train, y_train, save=True)

    rf_pipeline = rf_gs.best_estimator_
    svm_pipeline = svm_gs.best_estimator_

    # ------------------------------------------------------------------
    # Phase D: Ensemble weight selection
    # ------------------------------------------------------------------
    print("\n--- Selecting fusion weight (CV on training set) ---")
    best_w, best_cv_auc = select_fusion_weight(
        rf_pipeline, svm_pipeline, X_train, y_train
    )
    ensemble = EnsemblePredictor(
        rf_pipeline=rf_pipeline,
        svm_pipeline=svm_pipeline,
        fusion_weight=best_w,
        threshold=DECISION_THRESHOLD,
        cv_auc=best_cv_auc,
    )
    ensemble.save()

    # ------------------------------------------------------------------
    # Phase E: Evaluate on held-out test set
    # ------------------------------------------------------------------
    print("\n--- Evaluating on held-out test set ---")
    y_test_np = y_test.to_numpy()

    # Baseline
    y_pred_dummy = dummy.predict(X_test)

    # RF
    y_pred_rf = ensemble.predict_rf(X_test)
    y_prob_rf = ensemble.predict_proba_rf(X_test)

    # SVM
    y_pred_svm = ensemble.predict_svm(X_test)
    y_prob_svm = ensemble.predict_proba_svm(X_test)

    # Hybrid
    y_pred_hybrid = ensemble.predict_hybrid(X_test)
    y_prob_hybrid = ensemble.predict_proba_hybrid(X_test)

    # Metrics
    metrics = {
        "Baseline (Majority)": compute_metrics(y_test_np, y_pred_dummy),
        "Random Forest":       compute_metrics(y_test_np, y_pred_rf, y_prob_rf),
        "SVM":                 compute_metrics(y_test_np, y_pred_svm, y_prob_svm),
        "Hybrid (RF+SVM)":     compute_metrics(y_test_np, y_pred_hybrid, y_prob_hybrid),
    }

    table_df = comparison_table(metrics)

    # Save comparison table as CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = REPORTS_DIR / "comparison_table.csv"
    table_df.to_csv(table_path)
    print(f"Comparison table saved → {table_path}")

    # Classification reports
    print_classification_reports(
        y_test_np,
        {
            "Random Forest":   y_pred_rf,
            "SVM":             y_pred_svm,
            "Hybrid (RF+SVM)": y_pred_hybrid,
        },
    )

    # Confusion matrices
    plot_confusion_matrices(
        y_test_np,
        {
            "Random Forest":   y_pred_rf,
            "SVM":             y_pred_svm,
            "Hybrid (RF+SVM)": y_pred_hybrid,
        },
    )

    # ROC curves
    plot_roc_curves(
        y_test_np,
        {
            "Random Forest":   y_prob_rf,
            "SVM":             y_prob_svm,
            "Hybrid (RF+SVM)": y_prob_hybrid,
        },
    )

    # Calibration curves
    plot_calibration_curves(
        y_test_np,
        {
            "Random Forest":   y_prob_rf,
            "SVM":             y_prob_svm,
            "Hybrid (RF+SVM)": y_prob_hybrid,
        },
    )

    # Save run metadata
    meta = {
        "random_seed":          RANDOM_SEED,
        "test_size":            0.20,
        "fusion_weight":        best_w,
        "decision_threshold":   DECISION_THRESHOLD,
        "cv_auc_at_best_weight": best_cv_auc,
        "rf_best_params":       rf_gs.best_params_,
        "svm_best_params":      svm_gs.best_params_,
        "rf_cv_auc":            float(rf_gs.best_score_),
        "svm_cv_auc":           float(svm_gs.best_score_),
    }
    meta_path = REPORTS_DIR / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nRun metadata saved → {meta_path}")

    print("\n✅ Pipeline complete. Artifacts in: artifacts/  Reports in: reports/\n")


if __name__ == "__main__":
    run()
