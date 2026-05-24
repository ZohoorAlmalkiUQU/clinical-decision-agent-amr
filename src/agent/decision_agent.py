import os
from groq import Groq

MODEL_ID = "llama-3.3-70b-versatile"


class ClinicalDecisionAgent:
    """
    Multi-step reasoning agent that ranks antibiotics and explains recommendations.
    Uses Groq (free) to run LLaMA 3.3 70B for clinical reasoning.

    The confidence threshold is taken from the predictor so it matches the
    paper-optimal threshold for whichever ML model is in use (Table 4).
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def rank_antibiotics(self, resistance_scores: dict[str, float]) -> list[dict]:
        ranked = sorted(resistance_scores.items(), key=lambda x: x[1])
        return [{"antibiotic": ab, "resistance_probability": prob} for ab, prob in ranked]

    def is_confident(self, top_prob: float, threshold: float) -> bool:
        """Susceptible if resistance probability is BELOW the optimal threshold."""
        return top_prob < threshold

    def build_prompt(self, patient_context: dict, ranked: list[dict]) -> str:
        lines = [
            "You are a clinical decision support AI specialising in antimicrobial resistance (AMR).",
            "Based on the patient data and ML-predicted resistance probabilities below, "
            "provide a concise recommendation.",
            "",
            "## Patient Context",
        ]
        for k, v in patient_context.items():
            lines.append(f"- {k}: {v}")

        lines += [
            "",
            "## Antibiotic Resistance Predictions (lower = more likely susceptible)",
        ]
        for entry in ranked:
            lines.append(
                f"- {entry['antibiotic']}: {entry['resistance_probability']:.1%} resistance probability"
            )

        lines += [
            "",
            "Provide:",
            "1. Recommended antibiotic and clinical rationale.",
            "2. Confidence assessment.",
            "3. Any important clinical caveats or flags.",
            "Be concise (under 200 words).",
        ]
        return "\n".join(lines)

    def recommend(
        self,
        patient_context: dict,
        resistance_scores: dict[str, float],
        threshold: float = 0.28,          # paper-optimal LightGBM default; overridden by predictor
        shap_features: list[dict] | None = None,
    ) -> dict:
        ranked     = self.rank_antibiotics(resistance_scores)
        top        = ranked[0]
        confident  = self.is_confident(top["resistance_probability"], threshold)

        if not confident:
            return {
                "recommendation": None,
                "confidence":     "LOW",
                "explanation": (
                    f"All evaluated antibiotics show high resistance probability "
                    f"(best option: {top['antibiotic']} at {top['resistance_probability']:.1%}). "
                    "Manual clinical review and Infectious Disease consultation required."
                ),
                "ranked":         ranked,
                "shap_features":  shap_features,
            }

        prompt   = self.build_prompt(patient_context, ranked)
        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        explanation = response.choices[0].message.content

        return {
            "recommendation":       top["antibiotic"],
            "confidence":           "HIGH",
            "resistance_probability": top["resistance_probability"],
            "explanation":          explanation,
            "ranked":               ranked,
            "shap_features":        shap_features,
        }
