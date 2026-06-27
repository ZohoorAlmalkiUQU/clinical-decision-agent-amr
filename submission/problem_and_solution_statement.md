# Problem and Solution Statement

**Project:** AMR Clinical Decision Support System
**Team:** GROUP-6 — UQU CS6117 (Advanced Topics in AI)

---

Antimicrobial resistance (AMR) is one of the top global health threats, contributing to over a million deaths annually. A central driver is empirical prescribing: when a patient presents with a suspected bacterial infection, the treating clinician must choose an antibiotic *before* microbiology culture and sensitivity results return — often 48 to 72 hours later. Choosing an antibiotic the organism is resistant to delays effective treatment, worsens patient outcomes (especially in sepsis), and accelerates the spread of resistant strains through unnecessary broad-spectrum use.

Existing decision aids are largely static antibiograms — hospital-wide resistance percentages by organism and drug, updated annually. They do not account for an individual patient's comorbidities, recent labs, prior antibiotic exposure, or current clinical status, all of which materially affect the probability that a given organism-drug pairing will succeed.

**Our solution** is an AI-powered clinical decision support system that closes this gap by predicting *patient-specific* antibiotic resistance probabilities and turning them into an actionable, explainable recommendation — in seconds, at the point of care.

The system works in three stages:

1. **Prediction.** A clinician enters the patient's demographics, comorbidities, current labs (CBC, metabolic panel, lactate), vital signs, the organism identified on culture, and recent antibiotic exposure. A calibrated ensemble machine learning model (LightGBM, with XGBoost, CatBoost, and HistGradientBoosting also available) predicts the probability of resistance for each candidate antibiotic, using isotonic calibration so the probabilities are clinically meaningful confidence scores — not just rank scores.

2. **Explanation.** SHAP (TreeExplainer) decomposes the top prediction into the specific patient features driving it — for example, impaired renal clearance, recent fluoroquinolone exposure, or an elevated white cell count — so the recommendation is transparent and auditable rather than a black box.

3. **Reasoning and recommendation.** A clinical decision agent ranks all evaluated antibiotics from lowest to highest predicted resistance, then applies a confidence gate based on the paper-validated optimal threshold for the active model. If the top-ranked option is below the threshold, the agent invokes a large language model (LLaMA 3.3 70B) to generate a concise, doctor-readable rationale and caveats. If *no* antibiotic clears the threshold, the system deliberately withholds a recommendation and flags the case for Infectious Disease consultation — a built-in safety behaviour that prevents the model from fabricating confidence it doesn't have. The system also computes a qSOFA score live from entered vitals as an early sepsis-risk signal.

All models, thresholds, and calibration strategies are derived directly from our peer-reviewed research — *"Comparative Evaluation of Ensemble Machine Learning Models for Predicting Antibacterial Resistance from Electronic Health Records,"* published in [*Discover AI* (Springer Nature)](https://link.springer.com/article/10.1007/s44163-026-01436-4) — trained and validated on the Antibiotic Resistance Microbiology Dataset (997 patients, 1.2M+ observations, 20 organisms, 28 antibiotics) using patient-level cross-validation to prevent data leakage.

The result is a tool that gives frontline clinicians an evidence-based, explainable, patient-specific second opinion at the moment they need it most — supporting faster, more accurate empirical prescribing while preserving the clinician's final judgement and a clear escalation path to specialists when the evidence is insufficient.

*(Word count: ~470)*
