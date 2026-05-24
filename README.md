# AMR Clinical Decision Support System

> **Applied implementation of peer-reviewed research — accepted for publication in *Discover AI* (Springer Nature)**

An AI-powered clinical decision support system that predicts antibiotic resistance probabilities from patient EHR data, ranks treatment options, and generates explainable recommendations using a combination of ensemble machine learning and large language model reasoning.

---

## Research Paper

This system is the direct applied implementation of the following research:

> Almalki Z, Althagafi A, Al-Shareef S.
> **Comparative Evaluation of Ensemble Machine Learning Models for Predicting Antibacterial Resistance from Electronic Health Records.**
> *Discover AI*, Springer Nature. **Accepted — in press.**

All ML models, decision thresholds, calibration strategies, and performance benchmarks in this system are derived directly from the paper's methodology and results (Tables 3 & 4, Section 4.3).

---

## Overview

Antimicrobial resistance (AMR) is one of the most critical threats to global public health. Clinicians selecting antibiotics under uncertainty risk treatment failure and accelerate the spread of resistant organisms. Existing clinical tools rarely go beyond static antibiograms.

This project bridges that gap by combining:

- **Ensemble gradient boosting models** trained on de-identified EHR data to predict organism–antibiotic resistance probabilities
- **Isotonic probability calibration** to produce reliable confidence scores
- **SHAP explainability** so every prediction can be traced to specific patient features
- **LLaMA 3.3 70B (via Groq)** for agentic multi-step clinical reasoning and natural-language recommendation generation
- A **Streamlit clinical UI** that integrates all of the above into a usable decision-support tool

---

## Model Performance (Paper Table 4)

All four ensemble models were evaluated on the ARMD dataset using patient-level GroupKFold cross-validation with leakage-aware evaluation.

| Model | ROC-AUC | PR-AUC | Brier Score | F1 | Optimal Threshold |
| --- | --- | --- | --- | --- | --- |
| **LightGBM** ★ | **0.851** | **0.682** | **0.120** | **0.612** | 0.28 |
| HistGradientBoosting | 0.831 | 0.654 | 0.131 | 0.574 | 0.27 |
| XGBoost | 0.840 | 0.675 | 0.125 | 0.607 | 0.32 |
| CatBoost | 0.820 | 0.628 | 0.138 | 0.460 | 0.15 |

★ LightGBM is the best overall model and the default in the UI.

All models are calibrated using **isotonic regression** (`CalibratedClassifierCV`) and tuned with **Optuna** (20 trials per model). Optimal thresholds are determined by maximising F1 on the validation set (paper Section 4.3.5).

---

## System Architecture

```
Patient EHR Input (demographics, labs, vitals, microbiology, medications)
        │
        ▼
┌───────────────────────────────────────────────┐
│           ResistancePredictor                 │
│  • Calibrated ensemble model (LightGBM et al) │
│  • Predicts resistance prob. per antibiotic   │
│  • Paper-optimal threshold per model          │
│  • SHAP TreeExplainer (base model)            │
└───────────────────┬───────────────────────────┘
                    │  resistance scores + SHAP features
                    ▼
┌───────────────────────────────────────────────┐
│         ClinicalDecisionAgent                 │
│  • Ranks antibiotics (lowest resistance first) │
│  • Confidence gate: prob < optimal threshold? │
│  • If confident → LLaMA 3.3 70B (Groq)        │
│    generates clinical rationale               │
│  • If not confident → flags for manual review │
└───────────────────┬───────────────────────────┘
                    │
                    ▼
        Streamlit Clinical UI
  (recommendation · ranked table · SHAP · qSOFA)
```

---

## Key Features

- **Multi-model selection** — choose from all 4 calibrated ensemble models at runtime; performance metrics from the paper are shown inline
- **Per-model optimal thresholds** — confidence gates use the paper-validated threshold for whichever model is active, not a hardcoded value
- **SHAP feature importance** — top-8 feature contributions shown for the recommended antibiotic, using TreeExplainer on the uncalibrated base model
- **LLM clinical reasoning** — LLaMA 3.3 70B generates a concise, patient-specific rationale (under 200 words) via Groq's free inference API
- **Safe failure mode** — when all antibiotics exceed the resistance threshold, the system refuses to fabricate a recommendation and flags for Infectious Disease consultation
- **qSOFA sepsis screening** — automatically computed from entered vital signs
- **Dark-mode Streamlit UI** — full patient form with demographics, comorbidities, labs, vitals, and medication history

---

## Project Structure

```
clinical-decision-agent-amr/
├── demo_app/
│   └── app.py                   # Streamlit UI (main entry point)
├── src/
│   ├── agent/
│   │   ├── decision_agent.py    # ClinicalDecisionAgent — ranking + LLM reasoning
│   │   └── explainer.py         # SHAP utility functions
│   ├── inference/
│   │   └── predictor.py         # ResistancePredictor — loads models, runs SHAP
│   ├── model/
│   │   ├── lgbm_calibrated.pkl
│   │   ├── lgbm_final.pkl       # (uncalibrated, for SHAP)
│   │   ├── xgb_calibrated.pkl
│   │   ├── xgb_final.pkl
│   │   ├── cat_calibrated.pkl
│   │   ├── cat_final.pkl
│   │   ├── hgb_calibrated.pkl
│   │   ├── hgb_final.pkl
│   │   └── encoders/            # age_map, antibiotic_map, organism_map, feature_names
│   └── utils/
│       └── preprocessing.py     # Feature engineering & label encoding
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb  # Training, Optuna tuning, calibration, export
│   └── 03_agent_demo.ipynb
├── scripts/
│   └── export_encoders_from_abr_ml.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_patient.json      # Example patient for quick testing
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ZohoorAlmalkiUQU/clinical-decision-agent-amr.git
cd clinical-decision-agent-amr
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key (free at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Add model files

The trained model `.pkl` files are not tracked in git due to their size. Place the following files in `src/model/`:

```
lgbm_calibrated.pkl   lgbm_final.pkl
xgb_calibrated.pkl    xgb_final.pkl
cat_calibrated.pkl    cat_final.pkl
hgb_calibrated.pkl    hgb_final.pkl
```

To retrain from scratch, run `notebooks/02_model_training.ipynb`.

### 6. Run the app

```bash
streamlit run demo_app/app.py
```

---

## Dataset

The models were trained on the **Antibiotic Resistance Microbiology Dataset (ARMD)**:

| Property | Value |
| --- | --- |
| Patients | 997 |
| Observations | 1,213,641 |
| Organisms | 20 bacterial species |
| Antibiotics | 28 agents |
| EHR features | Labs, vitals, demographics, comorbidities, medication history |

All records are de-identified. No personally identifiable information (PII) is used or stored. The dataset includes microbiology cultures, CBC, metabolic panel, vital signs, prior antibiotic exposure, and Charlson-based comorbidity flags.

**Validation strategy:** Patient-level `GroupKFold` (k=5) to prevent data leakage — no patient appears in both train and validation splits.

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| ML models | LightGBM, XGBoost, CatBoost, scikit-learn HGB |
| Calibration | `CalibratedClassifierCV` (isotonic regression) |
| Hyperparameter tuning | Optuna (20 trials per model) |
| Explainability | SHAP (`TreeExplainer`) |
| LLM reasoning | LLaMA 3.3 70B via Groq API |
| UI | Streamlit |
| Data processing | pandas, NumPy, scikit-learn |
| Model serialisation | joblib |

---

## Citation

If you use this system or the underlying research in your work, please cite:

```text
Almalki Z, Althagafi A, Al-Shareef S.
Comparative Evaluation of Ensemble Machine Learning Models for Predicting
Antibacterial Resistance from Electronic Health Records.
Discover AI, Springer Nature. Accepted — in press.
```

---

## Team

**GROUP-6 — UQU CS6117 (Advanced Topics in AI)**
Umm Al-Qura University, Department of Computer Science

| Name | Role |
| --- | --- |
| Zohoor Almalki | Project lead, ML development, paper first author |
| Nada Alfhmi | Clinical domain research, evaluation |
| Wojood Almatrfi | System design, UI development |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> **Clinical disclaimer:** This system is a research prototype intended for academic and decision-support purposes only. It is not a substitute for clinical judgement, local antibiogram data, or the advice of a qualified Infectious Disease specialist. All treatment decisions remain the responsibility of the treating physician.
