"""
End-to-end pipeline: load patient JSON → predict resistance → agent recommendation.
Usage: python scripts/run_pipeline.py --patient data/sample_patient.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.inference.predictor import ResistancePredictor
from src.agent.decision_agent import ClinicalDecisionAgent

ANTIBIOTICS = [
    "Meropenem", "Piperacillin/Tazobactam", "Ceftriaxone", "Ciprofloxacin",
    "Gentamicin", "Ampicillin", "Trimethoprim/Sulfamethoxazole",
    "Cefazolin", "Levofloxacin", "Amikacin",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True, help="Path to patient JSON file")
    args = parser.parse_args()

    with open(args.patient) as f:
        patient = json.load(f)

    print("Patient:", patient)
    print()

    print("Running resistance predictions...")
    predictor = ResistancePredictor("lgbm_calibrated.pkl")
    resistance_scores = predictor.predict_all(patient, ANTIBIOTICS)

    print("Resistance probabilities:")
    for ab, prob in sorted(resistance_scores.items(), key=lambda x: x[1]):
        print(f"  {ab:<35} {prob:.3f}")
    print()

    print("Running agent reasoning...")
    agent = ClinicalDecisionAgent()
    result = agent.recommend(patient_context=patient, resistance_scores=resistance_scores)

    print("=== Clinical Decision Agent Result ===")
    print(f"Recommendation : {result['recommendation']}")
    print(f"Confidence     : {result['confidence']}")
    print(f"Explanation    :\n{result['explanation']}")


if __name__ == "__main__":
    main()
