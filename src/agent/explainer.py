import shap
import pandas as pd
import numpy as np


def compute_shap_values(model, features: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    return explainer(features)


def top_features(shap_values, feature_names: list, n: int = 5) -> list[dict]:
    """Return the top-n features driving the prediction."""
    mean_abs = np.abs(shap_values.values[0])
    indices = np.argsort(mean_abs)[::-1][:n]
    return [
        {"feature": feature_names[i], "shap_value": float(shap_values.values[0][i])}
        for i in indices
    ]
