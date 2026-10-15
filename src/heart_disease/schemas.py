"""
Input/output data contracts for the prediction service.

Uses plain dataclasses (no external validation library required).
The `PatientFeatures` class also performs all clinical range and
categorical-value checks so that `predict.py` stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from heart_disease.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DISCLAIMER,
    FEATURE_RANGES,
    FEATURE_VALID_VALUES,
    NUMERIC_FEATURES,
    RISK_TIER_HIGH_MIN,
    RISK_TIER_MODERATE_MIN,
)


def classify_risk_tier(probability: Optional[float]) -> dict:
    """Classify risk probability into 3 clinical stratification tiers:
    - Low Risk:       < 55%
    - Moderate Risk:  55% to 75%
    - High Risk:      > 75%
    """
    if probability is None:
        return {
            "tier": "Unknown",
            "label": "INDETERMINATE RISK",
            "badge": "UNKNOWN",
            "color": "#6c757d",
            "class_name": "risk-unknown",
            "recommendation": "Unable to calculate risk probability."
        }
    if probability < RISK_TIER_MODERATE_MIN:
        return {
            "tier": "Low Risk",
            "label": "LOW RISK / UNLIKELY SIGNIFICANT CAD",
            "badge": "LOW RISK (<55%)",
            "color": "#2a9d8f",
            "class_name": "risk-low",
            "recommendation": "Low likelihood of significant coronary stenosis (<50% narrowing). Routine preventative care and healthy lifestyle maintenance recommended."
        }
    elif probability <= RISK_TIER_HIGH_MIN:
        return {
            "tier": "Moderate Risk",
            "label": "MODERATE RISK / BORDERLINE ISCHEMIA",
            "badge": "MODERATE RISK (55% - 75%)",
            "color": "#f4a261",
            "class_name": "risk-moderate",
            "recommendation": "Intermediate probability zone (55% - 75%). Early or partial arterial narrowing may be present. Further non-invasive diagnostic evaluation recommended (e.g. Stress Echocardiogram, CT Coronary Calcium Score, specialized lipid panel)."
        }
    else:
        return {
            "tier": "High Risk",
            "label": "HIGH RISK / PROBABLE CORONARY ARTERY DISEASE",
            "badge": "HIGH RISK (>75%)",
            "color": "#e63946",
            "class_name": "risk-high",
            "recommendation": "High probability of significant multi-vessel or critical obstruction (>75%). Urgent cardiology consultation and consideration for invasive coronary angiography recommended."
        }



@dataclass
class PatientFeatures:
    """Validated input features for one patient.

    Attributes
    ----------
    age : int        Age in years.
    sex : int        0 = female, 1 = male.
    cp : int         Chest pain type (1–4).
    trestbps : int   Resting blood pressure (mm Hg).
    chol : int       Serum cholesterol (mg/dl).
    fbs : int        Fasting blood sugar > 120 mg/dl (0/1).
    restecg : int    Resting ECG results (0/1/2).
    thalach : int    Maximum heart rate achieved.
    exang : int      Exercise-induced angina (0/1).
    oldpeak : float  ST depression induced by exercise.
    slope : int      Slope of peak exercise ST segment (1/2/3).
    ca : int         Number of major vessels coloured by fluoroscopy (0–3).
    thal : int       Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect).
    """

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    def __post_init__(self) -> None:
        """Validate ranges and categorical values after construction."""
        errors: list[str] = []

        # --- numeric range checks ---
        numeric_vals: dict[str, float] = {
            "age":      float(self.age),
            "trestbps": float(self.trestbps),
            "chol":     float(self.chol),
            "thalach":  float(self.thalach),
            "oldpeak":  float(self.oldpeak),
        }
        for feat, val in numeric_vals.items():
            lo, hi = FEATURE_RANGES[feat]
            if not (lo <= val <= hi):
                errors.append(
                    f"'{feat}' = {val} is outside expected range [{lo}, {hi}]."
                )

        # --- categorical value checks ---
        categorical_vals: dict[str, int] = {
            "sex":     self.sex,
            "cp":      self.cp,
            "fbs":     self.fbs,
            "restecg": self.restecg,
            "exang":   self.exang,
            "slope":   self.slope,
            "ca":      self.ca,
            "thal":    self.thal,
        }
        for feat, val in categorical_vals.items():
            allowed = FEATURE_VALID_VALUES[feat]
            if val not in allowed:
                errors.append(
                    f"'{feat}' = {val} is not a valid value. Allowed: {sorted(allowed)}."
                )

        if errors:
            raise ValueError(
                "PatientFeatures validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

    def to_dict(self) -> dict[str, float]:
        """Return features as an ordered dict matching ALL_FEATURES column order."""
        raw = {
            "age":      self.age,
            "trestbps": self.trestbps,
            "chol":     self.chol,
            "thalach":  self.thalach,
            "oldpeak":  self.oldpeak,
            "sex":      self.sex,
            "cp":       self.cp,
            "fbs":      self.fbs,
            "restecg":  self.restecg,
            "exang":    self.exang,
            "slope":    self.slope,
            "ca":       self.ca,
            "thal":     self.thal,
        }
        # Return in canonical ALL_FEATURES order
        return {f: raw[f] for f in ALL_FEATURES}


@dataclass
class ModelPrediction:
    """Prediction result from a single model.

    Attributes
    ----------
    model : str           Identifier for the model (e.g. ``"random_forest"``).
    prediction : int      Binary prediction — 0 (no disease) or 1 (disease).
    probability : float | None
        Estimated probability of the positive class (disease).
        ``None`` when the model cannot produce calibrated probabilities.
    threshold : float | None
        Decision threshold applied to convert probability → prediction.
        ``None`` when probability is ``None``.
    disclaimer : str      Research-use disclaimer.
    """

    model: str
    prediction: int
    probability: Optional[float]
    threshold: Optional[float]
    risk_tier: Optional[str] = None
    disclaimer: str = field(default=DISCLAIMER, init=False)

    def __post_init__(self) -> None:
        if self.risk_tier is None and self.probability is not None:
            self.risk_tier = classify_risk_tier(self.probability)["tier"]

    def to_dict(self) -> dict:
        """Serialisable dictionary representation."""
        return {
            "model":       self.model,
            "prediction":  self.prediction,
            "probability": round(self.probability, 4) if self.probability is not None else None,
            "threshold":   self.threshold,
            "risk_tier":   self.risk_tier,
            "disclaimer":  self.disclaimer,
        }


@dataclass
class EnsemblePredictionResult:
    """Aggregated result containing all three model predictions.

    Attributes
    ----------
    random_forest : ModelPrediction
    svm : ModelPrediction
    hybrid : ModelPrediction
    fusion_weight : float
        The weight ``w`` used in ``P_hybrid = w * P_RF + (1-w) * P_SVM``.
    """

    random_forest: ModelPrediction
    svm: ModelPrediction
    hybrid: ModelPrediction
    fusion_weight: float

    def to_dict(self) -> dict:
        """Serialisable dictionary representation."""
        return {
            "random_forest":  self.random_forest.to_dict(),
            "svm":            self.svm.to_dict(),
            "hybrid":         self.hybrid.to_dict(),
            "fusion_weight":  self.fusion_weight,
        }
