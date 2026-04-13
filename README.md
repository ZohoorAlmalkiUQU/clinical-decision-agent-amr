# 🧠 Clinical Decision Agent for Antimicrobial Resistance (AMR)

An AI-powered clinical decision agent that predicts antibiotic resistance and provides context-aware treatment recommendations using patient data.

---

## 🚀 Overview

Antimicrobial resistance (AMR) is a major global health challenge that complicates treatment decisions and increases patient risk. Clinicians often need to select antibiotics under uncertainty, where incorrect choices can lead to treatment failure and the spread of resistant organisms.

This project introduces a **Clinical Decision Agent** that combines machine learning and agentic AI to support antibiotic selection. The system not only predicts resistance but also **reasons, ranks treatment options, and explains its decisions**, transforming predictive models into actionable clinical tools.

---

## 🎯 Key Features

- 🔍 **Antibiotic Resistance Prediction**
  - Predicts resistance probabilities for organism–antibiotic pairs using patient data

- 🤖 **Agentic AI Decision-Making**
  - Performs multi-step reasoning
  - Ranks treatment options
  - Applies confidence-based decision logic

- 🧠 **Context-Aware Recommendations**
  - Incorporates patient history, labs, vitals, and prior medications

- 📊 **Explainability**
  - Uses feature attribution (e.g., SHAP) to justify recommendations

- ⚠️ **Risk-Aware Decisions**
  - Flags uncertain cases instead of making unsafe recommendations

---

## 🏗️ System Architecture
```
[Any Patient Data Input]
↓
[ML Model (AMR Prediction)]
↓
[Clinical Decision Agent]
↓
[Reasoning + Ranking + Confidence Check]
↓
[Final Recommendation + Explanation]
```

---

## 🧠 Agent Design

This system follows **agentic AI principles**, acting as:

- **Decision Gatekeeper**
  - Determines whether a treatment decision is safe and reliable

- **Context-Aware Expert**
  - Adapts recommendations based on patient-specific clinical context

Unlike traditional models, this system **actively makes decisions and justifies them**, rather than only providing predictions.

---

## 📊 Dataset

This project is based on the **Antibiotic Resistance Microbiology Dataset (ARMD)**:
- De-identified electronic health records
- Includes:
  - Microbiology data
  - Laboratory results
  - Vital signs
  - Demographics
  - Prior medication exposure
  - Comorbidities

⚠️ **Note:** No personally identifiable information (PII) is used.

---

## ⚙️ Technologies Used

- **Machine Learning**
  - LightGBM / XGBoost
  - Scikit-learn

- **Explainability**
  - SHAP

- **Agentic AI Design**
  - Multi-step reasoning
  - Decision logic
  - Confidence thresholds

- **Optional Integration**
  - IBM watsonx Orchestrate (for workflow orchestration)

---

## 💻 Project Structure
```
├── data/
├── notebooks/
├── src/
│ ├── model/
│ ├── agent/
│ ├── inference/
│ └── utils/
├── demo_app/ # Optional (Streamlit UI)
├── scripts/
├── README.md
└── requirements.txt
```


---

## ▶️ Demo

### Example Workflow:

1. Input patient data (labs, vitals, history)
2. Model predicts resistance probabilities
3. Agent analyzes and ranks antibiotics
4. System outputs:
   - Recommended antibiotic
   - Confidence score
   - Explanation

---

## 📈 Impact

- Reduces cognitive load for clinicians
- Improves antibiotic selection accuracy
- Supports antimicrobial stewardship
- Helps reduce the spread of resistance

---

## 🔥 Key Contribution

> This project transforms traditional machine learning models into an **intelligent clinical agent** that not only predicts resistance but actively recommends and justifies treatment decisions.

---

## 🧪 Future Work

- Real-time clinical integration
- Multi-agent collaboration
- Temporal patient monitoring
- Reinforcement learning for treatment optimization

---

## 📜 License

MIT License

---
\
## 👥 Team

GROUP-6-UQU-CS6117  
- Zohoor Almalki.
- Nada Alfhmi.
- Wojood Almatrfi.

This project is developed by Master's students in Artificial Intelligence at Umm Al-Qura University. It reflects our academic training and research focus in applying advanced machine learning and agentic AI techniques to real-world healthcare challenges. Through this work, we aim to bridge the gap between theoretical AI concepts and practical clinical decision support systems, contributing to the development of intelligent, reliable, and impactful healthcare solutions.

---
