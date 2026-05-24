"""
AMR Resistance Predictor — inference-time wrapper.

The training used ONE unified model where `antibiotic` and `organism` are features.
To get resistance probabilities for multiple antibiotics, we run the same patient
row through the model once per antibiotic (varying the `antibiotic` code each time).
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import src.utils.preprocessing as _pre

_MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "model"

# Calibrated model filenames (used for probability prediction)
CALIBRATED_MODELS = {
    "LightGBM":           "lgbm_calibrated.pkl",
    "XGBoost":            "xgb_calibrated.pkl",
    "CatBoost":           "cat_calibrated.pkl",
    "HistGradientBoosting": "hgb_calibrated.pkl",
}

# Uncalibrated final model filenames (used for SHAP — TreeExplainer needs base tree)
FINAL_MODELS = {
    "LightGBM":           "lgbm_final.pkl",
    "XGBoost":            "xgb_final.pkl",
    "CatBoost":           "cat_final.pkl",
    "HistGradientBoosting": "hgb_final.pkl",
}

# Optimal decision thresholds from paper Table 4
OPTIMAL_THRESHOLDS = {
    "LightGBM":           0.28,
    "XGBoost":            0.32,
    "CatBoost":           0.15,
    "HistGradientBoosting": 0.27,
}

# Performance from paper Table 4 (for display in UI)
MODEL_PERFORMANCE = {
    "LightGBM":           {"ROC-AUC": 0.851, "PR-AUC": 0.682, "Brier": 0.120, "F1": 0.612},
    "XGBoost":            {"ROC-AUC": 0.840, "PR-AUC": 0.675, "Brier": 0.125, "F1": 0.607},
    "CatBoost":           {"ROC-AUC": 0.820, "PR-AUC": 0.628, "Brier": 0.138, "F1": 0.460},
    "HistGradientBoosting": {"ROC-AUC": 0.831, "PR-AUC": 0.654, "Brier": 0.131, "F1": 0.574},
}


class ResistancePredictor:
    """
    Loads a calibrated AMR model and predicts resistance probability
    for a patient across one or many antibiotics.
    Also exposes the uncalibrated base model for SHAP.
    """

    def __init__(self, model_name: str = "LightGBM"):
        if model_name not in CALIBRATED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(CALIBRATED_MODELS)}")
        self.model_name = model_name
        self.threshold   = OPTIMAL_THRESHOLDS[model_name]
        self.performance = MODEL_PERFORMANCE[model_name]

        calibrated_path = _MODEL_DIR / CALIBRATED_MODELS[model_name]
        self.model = joblib.load(calibrated_path)

        # Load uncalibrated model for SHAP (TreeExplainer requires the raw tree)
        final_path = _MODEL_DIR / FINAL_MODELS[model_name]
        self.base_model = joblib.load(final_path)

        _pre._load_encoders()

    def predict_for_antibiotic(self, patient: dict, antibiotic_name: str) -> float:
        """Return resistance probability (0–1) for a single antibiotic."""
        ab_map  = _pre._antibiotic_map
        ab_code = ab_map.get(antibiotic_name, ab_map.get("Other", 21))
        features = _pre.preprocess_patient(patient)
        features["antibiotic"] = ab_code
        return float(self.model.predict_proba(features)[:, 1][0])

    def predict_all(
        self,
        patient: dict,
        antibiotics: list[str] | None = None,
    ) -> dict[str, float]:
        """Return resistance probabilities for a list of antibiotics."""
        if antibiotics is None:
            antibiotics = list(_pre._antibiotic_map.keys())
        return {ab: self.predict_for_antibiotic(patient, ab) for ab in antibiotics}

    def explain_antibiotic(
        self,
        patient: dict,
        antibiotic_name: str,
        n_features: int = 8,
    ) -> list[dict]:
        """
        Return top-n SHAP feature contributions for one antibiotic prediction.
        Uses the uncalibrated base model with TreeExplainer.
        """
        import shap
        ab_map   = _pre._antibiotic_map
        ab_code  = ab_map.get(antibiotic_name, ab_map.get("Other", 21))
        features = _pre.preprocess_patient(patient)
        features["antibiotic"] = ab_code

        explainer   = shap.TreeExplainer(self.base_model)
        shap_values = explainer(features)

        # shap_values.values shape: (1, n_features) for binary — take class 1
        vals = shap_values.values[0]
        if vals.ndim == 2:          # multi-output: take resistant class
            vals = vals[:, 1]

        feat_names = list(features.columns)
        indices    = np.argsort(np.abs(vals))[::-1][:n_features]

        return [
            {
                "feature":    feat_names[i],
                "shap_value": float(vals[i]),
                "direction":  "increases resistance" if vals[i] > 0 else "decreases resistance",
            }
            for i in indices
        ]
