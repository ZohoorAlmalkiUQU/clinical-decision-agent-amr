"""
Inference-time preprocessing that mirrors the ABR_ML training pipeline.

Input  : raw patient dict (single encounter)
Output : a pd.DataFrame row with exactly the features the model expects

Encoders (organism_map.json, antibiotic_map.json, age_map.json, feature_names.json)
must be exported from ABR_ML once and placed in src/model/encoders/.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

_ENCODER_DIR = Path(__file__).resolve().parents[2] / "src" / "model" / "encoders"

# ── lazy-loaded encoder caches ──────────────────────────────────────────────
_organism_map: dict | None = None
_antibiotic_map: dict | None = None
_age_map: dict | None = None
_feature_names: list | None = None


def _load_encoders():
    global _organism_map, _antibiotic_map, _age_map, _feature_names
    if _organism_map is None:
        with open(_ENCODER_DIR / "organism_map.json") as f:
            _organism_map = json.load(f)
        with open(_ENCODER_DIR / "antibiotic_map.json") as f:
            _antibiotic_map = json.load(f)
        with open(_ENCODER_DIR / "age_map.json") as f:
            _age_map = json.load(f)
        with open(_ENCODER_DIR / "feature_names.json") as f:
            _feature_names = json.load(f)


# ── comorbidity grouping (mirrors 01_prepare_sample.ipynb) ──────────────────
def _map_comorbidity_to_group(label: str) -> str:
    s = label.lower()
    if "diabetes" in s:
        return "has_diabetes"
    if any(k in s for k in ["congestive heart failure", "heart failure", "chf"]):
        return "has_chf"
    if any(k in s for k in ["chronic kidney", "ckd", "renal failure", "nephritis", "nephrosis"]):
        return "has_ckd"
    if any(k in s for k in ["cancer", "malignant", "neoplasm", "leukemia", "lymphoma",
                              "myeloma", "metastatic", "sarcoma", "tumor"]):
        return "has_cancer"
    if "transplant" in s:
        return "has_transplant"
    if any(k in s for k in ["immunity disorder", "immunodeficiency", "hiv", "aids", "immunosuppress"]):
        return "has_immunosuppression"
    if any(k in s for k in ["copd", "emphysema", "chronic bronchitis", "bronchiectasis"]):
        return "has_copd"
    if any(k in s for k in ["cirrhosis", "hepatic", "hepatitis", "liver disease", "liver failure"]):
        return "has_liver_disease"
    if any(k in s for k in ["dementia", "alzheimer", "neurocognitive"]):
        return "has_dementia"
    if any(k in s for k in ["stroke", "cva", "cerebral infarction", "cerebrovascular"]):
        return "has_stroke"
    return "has_other_comorbid"


_COMORBIDITY_COLS = [
    "has_diabetes", "has_chf", "has_ckd", "has_cancer", "has_transplant",
    "has_immunosuppression", "has_copd", "has_liver_disease", "has_dementia",
    "has_stroke", "has_other_comorbid",
]


def preprocess_patient(patient: dict) -> pd.DataFrame:
    """
    Convert a raw patient dict into a model-ready DataFrame row.

    Expected patient keys
    ---------------------
    organism        : str  e.g. "ESCHERICHIA COLI"
    age             : str  e.g. "65-74 years" | "above 90"
    gender          : str  "M" / "F" / "male" / "female" / "0" / "1"
    comorbidities   : list[str]  e.g. ["Diabetes, complicated", "Renal failure"]
                      OR dict[str, int] of already-binarised flags

    Labs (all optional, use None/NaN if unavailable):
      median_wbc, median_neutrophils, median_lymphocytes, median_hgb, median_plt,
      median_na, median_hco3, median_bun, median_cr, median_lactate
      (Q25/Q75 also accepted; if only median provided, Q25=Q75=median)

    Vitals (all optional):
      median_heartrate, median_resprate, median_temp,
      median_sysbp, median_diasbp
      (Q25/Q75 also accepted)

    Medication features (all optional):
      med_within_30_days : bool
      n_med_categories   : int
      min_time_to_med    : int
      antibiotic_class   : int (encoded code)
      min_time_to_class  : int

    Period_Day : int (days since first culture in encounter, default 0)
    """
    _load_encoders()

    row: dict = {}

    # ── organism ──────────────────────────────────────────────────────────
    org_raw = str(patient.get("organism", "OTHER")).upper().strip()
    row["organism"] = _organism_map.get(org_raw, _organism_map.get("OTHER", 12))

    # ── demographics ──────────────────────────────────────────────────────
    age_raw = str(patient.get("age", "")).strip()
    row["age"] = _age_map.get(age_raw, np.nan)

    gender_raw = str(patient.get("gender", "")).strip().lower()
    gender_map_local = {"0": 0, "1": 1, "f": 0, "female": 0, "m": 1, "male": 1}
    row["gender"] = gender_map_local.get(gender_raw, np.nan)

    # ── comorbidities ─────────────────────────────────────────────────────
    for col in _COMORBIDITY_COLS:
        row[col] = 0

    comorbidities = patient.get("comorbidities", [])
    if isinstance(comorbidities, dict):
        for col in _COMORBIDITY_COLS:
            row[col] = int(comorbidities.get(col, 0))
    elif isinstance(comorbidities, (list, tuple)):
        for label in comorbidities:
            group = _map_comorbidity_to_group(str(label))
            row[group] = 1

    # ── labs ──────────────────────────────────────────────────────────────
    lab_vars = ["wbc", "neutrophils", "lymphocytes", "hgb", "plt",
                "na", "hco3", "bun", "cr", "lactate"]

    for v in lab_vars:
        med = patient.get(f"median_{v}", patient.get(v, np.nan))
        q25 = patient.get(f"Q25_{v}", med)
        q75 = patient.get(f"Q75_{v}", med)
        row[f"Q25_{v}"]    = _to_float32(q25)
        row[f"median_{v}"] = _to_float32(med)
        row[f"Q75_{v}"]    = _to_float32(q75)

    # ── vitals ────────────────────────────────────────────────────────────
    vital_vars = ["heartrate", "resprate", "temp", "sysbp", "diasbp"]
    for v in vital_vars:
        med = patient.get(f"median_{v}", patient.get(v, np.nan))
        q25 = patient.get(f"Q25_{v}", med)
        q75 = patient.get(f"Q75_{v}", med)
        row[f"Q25_{v}"]    = _to_float32(q25)
        row[f"median_{v}"] = _to_float32(med)
        row[f"Q75_{v}"]    = _to_float32(q75)

    # ── medication features ───────────────────────────────────────────────
    row["med_within_30_days"] = float(patient.get("med_within_30_days", np.nan))
    row["n_med_categories"]   = patient.get("n_med_categories", np.nan)
    row["min_time_to_med"]    = patient.get("min_time_to_med", np.nan)
    row["antibiotic_class"]   = patient.get("antibiotic_class", np.nan)
    row["min_time_to_class"]  = patient.get("min_time_to_class", np.nan)

    # ── period day ────────────────────────────────────────────────────────
    row["Period_Day"] = patient.get("Period_Day", 0)

    # ── build DataFrame and align to training feature order ───────────────
    df = pd.DataFrame([row])
    for col in _feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[_feature_names]

    # ── make_model_ready equivalent ───────────────────────────────────────
    df = df.replace({pd.NA: np.nan})
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str.startswith("Int") or dtype_str == "boolean":
            df[col] = df[col].astype("float32")

    return df


def _to_float32(val):
    try:
        return np.float32(val)
    except (TypeError, ValueError):
        return np.nan
