"""
Run this script ONCE from inside the ABR_ML project directory to export
the encoders and best model needed by the clinical-decision-agent.

Usage (from ABR_ML root):
    python path/to/export_encoders_from_abr_ml.py --out /path/to/clinical-decision-agent-amr/src/model

It reads the processed parquet files (new_sample_one_processed/) and the
saved calibrated model (models/lgbm_cal.pkl) that were produced by
01_prepare_sample.ipynb → 04_main_pipeline.ipynb.
"""
import argparse
import json
import os
import shutil
import joblib
import pandas as pd
from pathlib import Path


# ── reproduce the exact category codes from 01_prepare_sample.ipynb ─────────

ORGANISM_MIN_COUNT = 100
ANTIBIOTIC_MIN_COUNT = 100

AGE_ORDER = {
    "18-24 years": 0,
    "25-34 years": 1,
    "35-44 years": 2,
    "45-54 years": 3,
    "55-64 years": 4,
    "65-74 years": 5,
    "75-84 years": 6,
    "85-89 years": 7,
    "above 90":    8,
}


def build_organism_map(processed_folder: str) -> dict:
    """Rebuild organism name → int8 code mapping from the processed cohort."""
    raw_cohort = pd.read_parquet(
        os.path.join(processed_folder, "..", "new_sample_one", "cultures_cohort.parquet")
    )
    raw_cohort = raw_cohort[raw_cohort["susceptibility"].isin(["Resistant", "Susceptible"])].copy()

    counts = raw_cohort["organism"].value_counts()
    common = counts[counts >= ORGANISM_MIN_COUNT].index
    raw_cohort["organism"] = raw_cohort["organism"].apply(
        lambda x: x if x in common else "OTHER"
    )
    cat = pd.Categorical(raw_cohort["organism"])
    return {name: int(code) for name, code in zip(cat.categories, range(len(cat.categories)))}


def build_antibiotic_map(processed_folder: str) -> dict:
    """Rebuild antibiotic name → int8 code mapping."""
    raw_cohort = pd.read_parquet(
        os.path.join(processed_folder, "..", "new_sample_one", "cultures_cohort.parquet")
    )
    raw_cohort = raw_cohort[raw_cohort["susceptibility"].isin(["Resistant", "Susceptible"])].copy()

    counts = raw_cohort["antibiotic"].value_counts()
    common = counts[counts >= ANTIBIOTIC_MIN_COUNT].index
    raw_cohort["antibiotic"] = raw_cohort["antibiotic"].apply(
        lambda x: x if x in common else "Other"
    )
    cat = pd.Categorical(raw_cohort["antibiotic"])
    return {name: int(code) for name, code in zip(cat.categories, range(len(cat.categories)))}


def build_feature_names(merged_folder: str) -> list:
    """Extract the exact feature column list (X columns) from the merged dataset."""
    df = pd.read_parquet(os.path.join(merged_folder, "final_stage", "part.0.parquet"))

    ID_COLS = ["anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded", "order_time_jittered_utc"]
    TARGET = "susceptibility"

    feature_cols = [c for c in df.columns if c not in ID_COLS and c != TARGET]
    return feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        required=True,
        help="Destination: path/to/clinical-decision-agent-amr/src/model",
    )
    parser.add_argument(
        "--processed-folder",
        default="new_sample_one_processed",
        help="ABR_ML processed data folder (default: new_sample_one_processed)",
    )
    parser.add_argument(
        "--merged-folder",
        default="merged_sample_one_processed",
        help="ABR_ML merged data folder (default: merged_sample_one_processed)",
    )
    parser.add_argument(
        "--model",
        default="models/lgbm_cal.pkl",
        help="Path to calibrated LightGBM model pkl (default: models/lgbm_cal.pkl)",
    )
    args = parser.parse_args()

    out_model_dir = Path(args.out)
    encoders_dir = out_model_dir / "encoders"
    encoders_dir.mkdir(parents=True, exist_ok=True)

    print("Building organism map...")
    organism_map = build_organism_map(args.processed_folder)
    with open(encoders_dir / "organism_map.json", "w") as f:
        json.dump(organism_map, f, indent=2)
    print(f"  Saved {len(organism_map)} organism codes")

    print("Building antibiotic map...")
    antibiotic_map = build_antibiotic_map(args.processed_folder)
    with open(encoders_dir / "antibiotic_map.json", "w") as f:
        json.dump(antibiotic_map, f, indent=2)
    print(f"  Saved {len(antibiotic_map)} antibiotic codes")

    print("Saving age map...")
    with open(encoders_dir / "age_map.json", "w") as f:
        json.dump(AGE_ORDER, f, indent=2)

    print("Building feature names list...")
    feature_names = build_feature_names(args.merged_folder)
    with open(encoders_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Saved {len(feature_names)} feature names")

    print("Copying model...")
    dest_model = out_model_dir / "lgbm_calibrated.pkl"
    shutil.copy2(args.model, dest_model)
    print(f"  Copied to {dest_model}")

    print("\nDone. Files saved to:", out_model_dir)
    print("  encoders/organism_map.json")
    print("  encoders/antibiotic_map.json")
    print("  encoders/age_map.json")
    print("  encoders/feature_names.json")
    print("  lgbm_calibrated.pkl")


if __name__ == "__main__":
    main()
