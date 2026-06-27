import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, date
from src.inference.predictor import (
    ResistancePredictor, CALIBRATED_MODELS, MODEL_PERFORMANCE, OPTIMAL_THRESHOLDS
)
from src.agent.decision_agent import ClinicalDecisionAgent

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── layout ─────────────────────────────────────────────────── */
    .block-container { padding-top: 2.9rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; }

    /* give tabs breathing room so the first tab label is never clipped */
    .stTabs { margin-top: 10px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding-top: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 600;
        color: #a8c4d8 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a5276 !important;
        color: #ffffff !important;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 12px; }

    /* ── header ─────────────────────────────────────────────────── */
    .clinical-header {
        background: linear-gradient(135deg, #0d2b4e 0%, #1a5276 100%);
        color: #ffffff !important;
        padding: 18px 28px;
        border-radius: 10px;
        margin-bottom: 22px;
    }
    .clinical-header * { color: #ffffff !important; }

    /* ── section headers ─────────────────────────────────────────── */
    .section-header {
        color: #5dade2 !important;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 2px solid #1a5276;
        padding-bottom: 6px;
        margin: 18px 0 12px 0;
    }

    /* ── helper labels ───────────────────────────────────────────── */
    .ref-range {
        font-size: 11px;
        color: #8899aa !important;
        margin-top: -12px;
        margin-bottom: 8px;
    }
    .vitals-flag {
        font-size: 12px;
        margin-top: -10px;
        margin-bottom: 6px;
    }

    /* ── recommendation cards ────────────────────────────────────── */
    .rec-high {
        background: linear-gradient(135deg, #0d3320, #1a4a2e);
        border: 2px solid #27ae60;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
    }
    .rec-high * { color: #a3e4c0 !important; }

    .rec-low {
        background: linear-gradient(135deg, #2d0f0f, #4a1a1a);
        border: 2px solid #c0392b;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
    }
    .rec-low * { color: #f5b7b1 !important; }

    /* ── reasoning / notes boxes ─────────────────────────────────── */
    .reasoning-box {
        background: #151e2d !important;
        border-left: 4px solid #4a9eda;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
        line-height: 1.65;
        margin-top: 12px;
        color: #d6e8f5 !important;
    }

    .disclaimer-box {
        background: #1a1500 !important;
        border: 1px solid #c8960a;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 12px;
        color: #f5d76e !important;
        margin-top: 14px;
    }
    .disclaimer-box * { color: #f5d76e !important; }

    /* ── report header ───────────────────────────────────────────── */
    .report-header {
        background: #0d2b4e;
        padding: 14px 24px;
        border-radius: 8px;
        margin: 20px 0 18px 0;
        color: #ffffff !important;
    }
    .report-header * { color: #ffffff !important; }

    /* ── qSOFA ───────────────────────────────────────────────────── */
    .qsofa-box {
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        color: #ffffff !important;
    }
    .qsofa-box * { color: #ffffff !important; }

    /* ── footer ──────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #8899aa !important;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #2a3244;
    }
    .footer * { color: #8899aa !important; }
</style>
""", unsafe_allow_html=True)

# ── cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_predictor(model_name: str):
    return ResistancePredictor(model_name)

@st.cache_resource
def get_agent():
    return ClinicalDecisionAgent()

# ── constants ─────────────────────────────────────────────────────────────────
ALL_ORGANISMS = [
    "ESCHERICHIA COLI", "KLEBSIELLA PNEUMONIAE", "KLEBSIELLA OXYTOCA",
    "PSEUDOMONAS AERUGINOSA", "MUCOID PSEUDOMONAS AERUGINOSA",
    "STAPHYLOCOCCUS AUREUS", "STAPH AUREUS {MRSA}",
    "ENTEROCOCCUS FAECALIS", "ENTEROCOCCUS SPECIES",
    "ENTEROBACTER CLOACAE COMPLEX", "PROTEUS MIRABILIS",
    "SERRATIA MARCESCENS", "CITROBACTER FREUNDII COMPLEX",
    "CITROBACTER KOSERI", "MORGANELLA MORGANII",
    "PROVIDENCIA RETTGERI", "COAG NEGATIVE STAPHYLOCOCCUS",
    "STREPTOCOCCUS AGALACTIAE (GROUP B)", "ACHROMOBACTER XYLOSOXIDANS", "OTHER",
]

INFECTION_SITES = [
    "Blood (Bacteremia / Sepsis)",
    "Urinary Tract (UTI / Pyelonephritis)",
    "Respiratory (Pneumonia / HAP / VAP)",
    "Intra-abdominal / Peritonitis",
    "Wound / Surgical Site Infection",
    "Central Line (CLABSI)",
    "Bone / Joint (Osteomyelitis / Septic Arthritis)",
    "CNS (Meningitis / Ventriculitis)",
    "Skin / Soft Tissue (SSTI / Cellulitis)",
    "Other / Unknown",
]

ANTIBIOTIC_CLASSES = {
    "Carbapenems": ["Meropenem", "Imipenem", "Ertapenem"],
    "Penicillins & β-lactam combos": [
        "Ampicillin", "Ampicillin/Sulbactam",
        "Amoxicillin/Clavulanic Acid", "Piperacillin/Tazobactam", "Penicillin",
    ],
    "Cephalosporins": ["Cefazolin", "Cefepime", "Cefoxitin", "Ceftazidime", "Ceftriaxone"],
    "Fluoroquinolones": ["Ciprofloxacin", "Levofloxacin", "Moxifloxacin"],
    "Aminoglycosides": ["Amikacin", "Gentamicin", "Tobramycin"],
    "Glycopeptides & Oxazolidinones": ["Vancomycin", "Linezolid"],
    "Monobactams": ["Aztreonam"],
    "Other": [
        "Clindamycin", "Erythromycin", "Nitrofurantoin",
        "Oxacillin", "Tetracycline", "Trimethoprim/Sulfamethoxazole",
    ],
}

DEFAULT_ANTIBIOTICS = {
    "Meropenem", "Imipenem", "Ertapenem",
    "Piperacillin/Tazobactam", "Ceftriaxone", "Cefepime",
    "Ciprofloxacin", "Levofloxacin",
    "Amikacin", "Gentamicin",
    "Vancomycin", "Trimethoprim/Sulfamethoxazole",
    "Ampicillin", "Cefazolin",
}

AB_CLASS_MAP = {
    "None": 0, "Beta-lactam / Penicillin": 1, "Fluoroquinolone": 2,
    "Aminoglycoside": 3, "Glycopeptide": 4, "Carbapenem": 5,
    "Macrolide": 6, "Other": 7,
}

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="clinical-header">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <div>
      <div style="font-size:22px; font-weight:700; margin-bottom:4px;">
        🏥 &nbsp;Antimicrobial Resistance — Clinical Decision Support
      </div>
      <div style="opacity:.85; font-size:13px;">
        Ensemble ML (Gradient Boosting) + LLaMA 3.3 (Groq) + SHAP Explainability
      </div>
      <div style="opacity:.7; font-size:11px; margin-top:6px;">
        Based on: <em>Comparative Evaluation of Ensemble ML Models for Predicting Antibacterial Resistance from EHR</em>
        &nbsp;— <strong>Discover AI</strong> (Springer Nature, 2026) —
        <a href="https://link.springer.com/article/10.1007/s44163-026-01436-4" target="_blank" style="color:#5dade2;">read paper</a>
      </div>
    </div>
    <div style="text-align:right; font-size:12px; opacity:.8; line-height:1.6;">
      GROUP-6-UQU-CS6117<br>Umm Al-Qura University<br>
      Advanced Topics in AI — CS6117
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── architecture & credits links ────────────────────────────────────────────
_arch_col, _credits_col = st.columns(2)
with _arch_col:
    with st.expander("🏗️  System Architecture", expanded=False):
        with open(os.path.join(os.path.dirname(__file__), "architecture.html"), encoding="utf-8") as f:
            components.html(f.read(), height=700, scrolling=True)
with _credits_col:
    with st.expander("👥  Team & Credits", expanded=False):
        with open(os.path.join(os.path.dirname(__file__), "credits.html"), encoding="utf-8") as f:
            components.html(f.read(), height=700, scrolling=True)

# ── model selector panel ──────────────────────────────────────────────────────
with st.expander("⚙️  ML Model Selection & Performance (from published research)", expanded=False):
    st.markdown(
        "Select the gradient boosting model to use for resistance prediction. "
        "All models were trained on the ARMD dataset (997 patients, 1.2M observations) "
        "with patient-level GroupKFold CV, isotonic calibration, and Optuna tuning. "
        "**LightGBM** is the best overall model per paper Table 4."
    )
    st.info(
        "📄 **Paper:** Almalki Z, Althagafi A, Al-Shareef S. "
        "*Comparative Evaluation of Ensemble Machine Learning Models for Predicting "
        "Antibacterial Resistance from Electronic Health Records.* "
        "**Discover AI** (Springer Nature, 2026). "
        "[Read the paper](https://link.springer.com/article/10.1007/s44163-026-01436-4).",
        icon=None,
    )
    _model_cols = st.columns(4)
    _perf_labels = {"ROC-AUC": "🎯", "PR-AUC": "📊", "Brier": "📉", "F1": "⚖️"}
    for _i, (_mname, _mperf) in enumerate(MODEL_PERFORMANCE.items()):
        with _model_cols[_i]:
            _badge = "🏆 " if _mname == "LightGBM" else ""
            st.markdown(f"**{_badge}{_mname}**")
            for _metric, _icon in _perf_labels.items():
                st.markdown(f"{_icon} {_metric}: `{_mperf[_metric]}`")
            st.caption(f"Opt. threshold: `{OPTIMAL_THRESHOLDS[_mname]}`")

    selected_model = st.selectbox(
        "Active prediction model",
        options=list(CALIBRATED_MODELS.keys()),
        index=0,
        help="LightGBM achieved the highest ROC-AUC (0.851) and PR-AUC (0.682) in the paper.",
    )
    _sel_perf = MODEL_PERFORMANCE[selected_model]
    st.info(
        f"**{selected_model}** — ROC-AUC {_sel_perf['ROC-AUC']} · "
        f"PR-AUC {_sel_perf['PR-AUC']} · Brier {_sel_perf['Brier']} · "
        f"Optimal threshold {OPTIMAL_THRESHOLDS[selected_model]}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PATIENT ENTRY FORM
# ══════════════════════════════════════════════════════════════════════════════
with st.form("patient_form"):

    tab_demo, tab_micro, tab_labs, tab_vitals, tab_meds = st.tabs([
        "👤  Patient & Demographics",
        "🔬  Microbiology & Culture",
        "🧪  Laboratory Results",
        "💓  Vital Signs",
        "💊  Medications & History",
    ])

    # ── TAB 1 — Demographics ──────────────────────────────────────────────────
    with tab_demo:
        st.markdown('<div class="section-header">Patient Identification</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_id   = st.text_input("Patient ID / MRN", placeholder="e.g., PT-2026-0042")
            case_date    = st.date_input("Case / Admission Date", value=date.today())
        with c2:
            ward         = st.text_input("Ward / Unit", placeholder="e.g., ICU, Medical Ward 3")
            attending    = st.text_input("Attending Physician", placeholder="Dr. ...")
        with c3:
            age          = st.selectbox("Age Group", [
                "18-24 years", "25-34 years", "35-44 years", "45-54 years",
                "55-64 years", "65-74 years", "75-84 years", "85-89 years", "above 90",
            ], index=5)
            gender       = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)

        st.markdown('<div class="section-header">Comorbidities</div>', unsafe_allow_html=True)
        ca, cb, cc = st.columns(3)
        with ca:
            has_diabetes    = st.checkbox("Diabetes mellitus (complicated)")
            has_ckd         = st.checkbox("Chronic kidney disease (CKD)")
            has_chf         = st.checkbox("Congestive heart failure (CHF)")
        with cb:
            has_cancer      = st.checkbox("Active cancer / malignancy")
            has_copd        = st.checkbox("COPD / chronic pulmonary disease")
            has_liver       = st.checkbox("Liver disease / cirrhosis")
        with cc:
            has_transplant  = st.checkbox("Organ transplant / immunosuppression")
            has_stroke      = st.checkbox("Stroke / cerebrovascular disease")
            has_dementia    = st.checkbox("Dementia / neurocognitive disorder")

    # ── TAB 2 — Microbiology ──────────────────────────────────────────────────
    with tab_micro:
        st.markdown('<div class="section-header">Culture & Sensitivity Details</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            organism       = st.selectbox("Identified Organism (Culture Result)", ALL_ORGANISMS)
            infection_site = st.selectbox("Infection Site / Source", INFECTION_SITES)
        with m2:
            culture_date   = st.date_input("Culture Collection Date", value=date.today())
            period_day     = st.number_input(
                "Days Since First Positive Culture", min_value=0, max_value=365, value=1, step=1,
            )

        st.markdown('<div class="section-header">Antibiotic Panel to Evaluate</div>', unsafe_allow_html=True)
        st.caption(
            "Select the antibiotics to include in the resistance analysis. "
            "Pre-selected panel covers the most clinically relevant options."
        )

        selected_antibiotics: list[str] = []
        class_cols = st.columns(4)
        for idx, (cls_name, ab_list) in enumerate(ANTIBIOTIC_CLASSES.items()):
            with class_cols[idx % 4]:
                st.markdown(f"**{cls_name}**")
                for ab in ab_list:
                    if st.checkbox(ab, value=(ab in DEFAULT_ANTIBIOTICS), key=f"ab__{ab}"):
                        selected_antibiotics.append(ab)

    # ── TAB 3 — Labs ─────────────────────────────────────────────────────────
    with tab_labs:
        st.markdown('<div class="section-header">Complete Blood Count (CBC)</div>', unsafe_allow_html=True)
        l1, l2, l3, l4, l5 = st.columns(5)
        with l1:
            wbc          = st.number_input("WBC (×10³/µL)",       value=10.0, min_value=0.0, max_value=200.0, step=0.5)
            st.markdown('<p class="ref-range">Normal: 4.5–11.0</p>', unsafe_allow_html=True)
        with l2:
            neutrophils  = st.number_input("Neutrophils (×10³/µL)", value=7.0, min_value=0.0, max_value=150.0, step=0.5)
            st.markdown('<p class="ref-range">Normal: 1.8–7.7</p>', unsafe_allow_html=True)
        with l3:
            lymphocytes  = st.number_input("Lymphocytes (×10³/µL)", value=1.5, min_value=0.0, max_value=30.0, step=0.5)
            st.markdown('<p class="ref-range">Normal: 1.0–4.8</p>', unsafe_allow_html=True)
        with l4:
            hgb          = st.number_input("Hemoglobin (g/dL)",    value=12.0, min_value=0.0, max_value=25.0, step=0.5)
            st.markdown('<p class="ref-range">Normal: 12–17.5</p>', unsafe_allow_html=True)
        with l5:
            plt          = st.number_input("Platelets (×10³/µL)", value=200.0, min_value=0.0, max_value=2000.0, step=10.0)
            st.markdown('<p class="ref-range">Normal: 150–400</p>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Metabolic / Chemistry Panel</div>', unsafe_allow_html=True)
        lm1, lm2, lm3, lm4, lm5 = st.columns(5)
        with lm1:
            na           = st.number_input("Sodium (mEq/L)",       value=138.0, min_value=100.0, max_value=180.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 135–145</p>', unsafe_allow_html=True)
        with lm2:
            hco3         = st.number_input("Bicarbonate (mEq/L)",  value=22.0, min_value=5.0, max_value=50.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 22–29</p>', unsafe_allow_html=True)
        with lm3:
            bun          = st.number_input("BUN (mg/dL)",          value=15.0, min_value=0.0, max_value=300.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 7–25</p>', unsafe_allow_html=True)
        with lm4:
            cr           = st.number_input("Creatinine (mg/dL)",   value=1.0, min_value=0.1, max_value=30.0, step=0.1)
            st.markdown('<p class="ref-range">Normal: 0.6–1.2</p>', unsafe_allow_html=True)
        with lm5:
            lactate      = st.number_input("Lactate (mmol/L)",     value=1.5, min_value=0.0, max_value=30.0, step=0.1)
            st.markdown('<p class="ref-range">Normal: 0.5–2.0</p>', unsafe_allow_html=True)

    # ── TAB 4 — Vitals ────────────────────────────────────────────────────────
    with tab_vitals:
        st.markdown('<div class="section-header">Current Vital Signs</div>', unsafe_allow_html=True)
        v1, v2, v3, v4, v5 = st.columns(5)
        with v1:
            temp         = st.number_input("Temperature (°C)", value=37.5, min_value=30.0, max_value=43.0, step=0.1)
            st.markdown('<p class="ref-range">Normal: 36.5–37.5°C</p>', unsafe_allow_html=True)
            if temp >= 38.3:
                st.markdown('<p class="vitals-flag" style="color:#c0392b;">🔴 Fever</p>', unsafe_allow_html=True)
            elif temp >= 37.8:
                st.markdown('<p class="vitals-flag" style="color:#d68910;">🟡 Low-grade fever</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="vitals-flag" style="color:#1e8449;">🟢 Normal</p>', unsafe_allow_html=True)
        with v2:
            heartrate    = st.number_input("Heart Rate (bpm)",  value=80.0, min_value=20.0, max_value=300.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 60–100 bpm</p>', unsafe_allow_html=True)
            if heartrate > 100:
                st.markdown('<p class="vitals-flag" style="color:#c0392b;">🔴 Tachycardia</p>', unsafe_allow_html=True)
            elif heartrate < 60:
                st.markdown('<p class="vitals-flag" style="color:#c0392b;">🔴 Bradycardia</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="vitals-flag" style="color:#1e8449;">🟢 Normal</p>', unsafe_allow_html=True)
        with v3:
            sysbp        = st.number_input("Systolic BP (mmHg)", value=120.0, min_value=40.0, max_value=280.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 90–120 mmHg</p>', unsafe_allow_html=True)
            if sysbp < 90:
                st.markdown('<p class="vitals-flag" style="color:#c0392b;">🔴 Hypotension</p>', unsafe_allow_html=True)
            elif sysbp > 140:
                st.markdown('<p class="vitals-flag" style="color:#d68910;">🟡 Hypertension</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="vitals-flag" style="color:#1e8449;">🟢 Normal</p>', unsafe_allow_html=True)
        with v4:
            diasbp       = st.number_input("Diastolic BP (mmHg)", value=75.0, min_value=20.0, max_value=180.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 60–80 mmHg</p>', unsafe_allow_html=True)
        with v5:
            resprate     = st.number_input("Resp. Rate (/min)", value=16.0, min_value=4.0, max_value=80.0, step=1.0)
            st.markdown('<p class="ref-range">Normal: 12–20/min</p>', unsafe_allow_html=True)
            if resprate > 20:
                st.markdown('<p class="vitals-flag" style="color:#c0392b;">🔴 Tachypnea</p>', unsafe_allow_html=True)
            elif resprate < 12:
                st.markdown('<p class="vitals-flag" style="color:#d68910;">🟡 Bradypnea</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="vitals-flag" style="color:#1e8449;">🟢 Normal</p>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Sepsis Screening — qSOFA</div>', unsafe_allow_html=True)
        qsofa_ms = st.checkbox("Altered mental status (new confusion / GCS < 15)?")
        qsofa_rr = 1 if resprate >= 22 else 0
        qsofa_bp = 1 if sysbp    <= 100 else 0
        qsofa_ms_val = 1 if qsofa_ms else 0
        qsofa_total  = qsofa_rr + qsofa_bp + qsofa_ms_val

        sq1, sq2 = st.columns([1, 3])
        with sq1:
            bg = "#c0392b" if qsofa_total >= 2 else ("#d68910" if qsofa_total == 1 else "#1e8449")
            st.markdown(f"""
            <div class="qsofa-box" style="background:{bg}; color:white;">
                <div style="font-size:36px; font-weight:800; line-height:1;">{qsofa_total}</div>
                <div style="font-size:11px; opacity:.9;">/ 3 &nbsp;qSOFA</div>
            </div>
            """, unsafe_allow_html=True)
        with sq2:
            criteria = []
            if qsofa_rr:    criteria.append("RR ≥ 22/min ✓")
            if qsofa_bp:    criteria.append("Systolic BP ≤ 100 mmHg ✓")
            if qsofa_ms_val: criteria.append("Altered mental status ✓")
            if qsofa_total >= 2:
                st.error(f"**High qSOFA ({qsofa_total}/3)** — Elevated risk of sepsis-related organ failure.  \n"
                         f"Criteria met: {', '.join(criteria)}.  \nConsider ICU level of care and blood cultures × 2.")
            elif qsofa_total == 1:
                st.warning(f"**Moderate qSOFA ({qsofa_total}/3)** — Monitor closely.  \n"
                           f"Criteria met: {', '.join(criteria)}.")
            else:
                st.success("**Low qSOFA (0/3)** — No immediate criteria for sepsis-related poor outcome met.")

    # ── TAB 5 — Medications ───────────────────────────────────────────────────
    with tab_meds:
        st.markdown('<div class="section-header">Prior Antibiotic Exposure (Last 30 Days)</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            med_within_30    = st.checkbox("Antibiotics administered in the last 30 days?")
            n_meds           = st.number_input("Number of antibiotic categories used", min_value=0, max_value=10, value=0, step=1)
        with p2:
            antibiotic_class = st.selectbox("Most recent antibiotic class", list(AB_CLASS_MAP.keys()))
            min_time_to_med  = st.number_input("Days from culture to first antibiotic given", min_value=0, max_value=365, value=0, step=1)
        with p3:
            min_time_to_cls  = st.number_input("Days from culture to most recent class", min_value=0, max_value=365, value=0, step=1)

        st.markdown('<div class="section-header">Additional Clinical Notes (Optional)</div>', unsafe_allow_html=True)
        clinical_notes = st.text_area(
            "Additional context for the AI agent",
            placeholder=(
                "e.g., patient transferred from another facility, prior resistance history, "
                "known drug allergies, recent invasive procedure..."
            ),
            height=90,
            label_visibility="collapsed",
        )

    # ── submit button ─────────────────────────────────────────────────────────
    st.markdown("---")
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        submitted = st.form_submit_button(
            "🔬  Analyze Patient & Get Recommendation",
            type="primary",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    if not selected_antibiotics:
        st.error("⚠️ Please select at least one antibiotic in the **Microbiology** tab before running the analysis.")
        st.stop()

    # build comorbidities list
    comorbidities = []
    if has_diabetes:   comorbidities.append("Diabetes, complicated")
    if has_ckd:        comorbidities.append("Chronic kidney disease")
    if has_chf:        comorbidities.append("Congestive heart failure")
    if has_cancer:     comorbidities.append("Cancer")
    if has_copd:       comorbidities.append("COPD")
    if has_liver:      comorbidities.append("Liver disease")
    if has_transplant: comorbidities.append("Organ transplant status")
    if has_stroke:     comorbidities.append("Stroke")
    if has_dementia:   comorbidities.append("Dementia")

    patient = {
        "organism":          organism,
        "age":               age,
        "gender":            "F" if gender == "Female" else "M",
        "comorbidities":     comorbidities,
        "median_wbc":        wbc,
        "median_neutrophils": neutrophils,
        "median_lymphocytes": lymphocytes,
        "median_hgb":        hgb,
        "median_plt":        plt,
        "median_na":         na,
        "median_hco3":       hco3,
        "median_bun":        bun,
        "median_cr":         cr,
        "median_lactate":    lactate,
        "median_heartrate":  heartrate,
        "median_resprate":   resprate,
        "median_temp":       temp,
        "median_sysbp":      sysbp,
        "median_diasbp":     diasbp,
        "med_within_30_days": med_within_30,
        "n_med_categories":  int(n_meds),
        "min_time_to_med":   int(min_time_to_med),
        "antibiotic_class":  AB_CLASS_MAP.get(antibiotic_class, 0),
        "min_time_to_class": int(min_time_to_cls),
        "Period_Day":        int(period_day),
    }

    with st.spinner(f"Running {selected_model} resistance prediction model..."):
        predictor = get_predictor(selected_model)
        resistance_scores = predictor.predict_all(patient, selected_antibiotics)

    # SHAP explanation for the top-ranked (lowest resistance) antibiotic
    shap_features = None
    top_antibiotic_for_shap = min(resistance_scores, key=resistance_scores.get)
    with st.spinner(f"Computing SHAP feature importance for {top_antibiotic_for_shap}..."):
        try:
            shap_features = predictor.explain_antibiotic(patient, top_antibiotic_for_shap, n_features=8)
        except Exception:
            shap_features = None  # SHAP is non-critical; continue without it

    with st.spinner("AI agent reasoning and generating clinical recommendation..."):
        agent = get_agent()
        result = agent.recommend(
            patient, resistance_scores,
            threshold=predictor.threshold,
            shap_features=shap_features,
        )

    # ── report header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="report-header">
      <div style="font-size:18px; font-weight:700;">📋 &nbsp;Clinical Analysis Report</div>
      <div style="font-size:12px; opacity:.8; margin-top:5px; line-height:1.8;">
        <span>Patient: <strong>{patient_id or "—"}</strong></span> &nbsp;|&nbsp;
        <span>Ward: <strong>{ward or "—"}</strong></span> &nbsp;|&nbsp;
        <span>Organism: <strong>{organism}</strong></span> &nbsp;|&nbsp;
        <span>Site: <strong>{infection_site}</strong></span> &nbsp;|&nbsp;
        <span>Generated: <strong>{datetime.now().strftime("%Y-%m-%d  %H:%M")}</strong></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_rec, col_table = st.columns([5, 7])

    # ── recommendation panel ──────────────────────────────────────────────────
    with col_rec:
        if result["recommendation"]:
            res_pct = result["resistance_probability"]
            st.markdown(f"""
            <div class="rec-high">
              <div style="font-size:11px; font-weight:700; color:#6ecf97;
                          text-transform:uppercase; letter-spacing:1px;">
                ✅ &nbsp;Recommended Antibiotic
              </div>
              <div style="font-size:30px; font-weight:800; color:#a3e4c0; margin:10px 0 6px;">
                {result["recommendation"]}
              </div>
              <div style="font-size:14px; color:#7ecfa5;">
                Predicted resistance probability: &nbsp;<strong>{res_pct:.1%}</strong>
              </div>
              <div style="margin-top:12px; display:inline-block; padding:5px 14px;
                          background:#1e6e3e; color:#a3e4c0; border-radius:20px;
                          font-size:12px; font-weight:700; letter-spacing:.5px;">
                HIGH CONFIDENCE
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            top_ab   = result["ranked"][0]["antibiotic"]
            top_prob = result["ranked"][0]["resistance_probability"]
            st.markdown(f"""
            <div class="rec-low">
              <div style="font-size:11px; font-weight:700; color:#f1a9a0;
                          text-transform:uppercase; letter-spacing:1px;">
                ⚠️ &nbsp;Manual Review Required
              </div>
              <div style="font-size:18px; font-weight:700; color:#f5b7b1; margin:10px 0 6px;">
                All evaluated antibiotics show high resistance probability
              </div>
              <div style="font-size:13px; color:#e88380;">
                Best option: <strong>{top_ab}</strong> at <strong>{top_prob:.1%}</strong> resistance
              </div>
              <div style="margin-top:12px; display:inline-block; padding:5px 14px;
                          background:#7b1c1c; color:#f5b7b1; border-radius:20px;
                          font-size:12px; font-weight:700; letter-spacing:.5px;">
                LOW CONFIDENCE — CONSULT INFECTIOUS DISEASE
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**AI Clinical Reasoning**")
        st.markdown(f'<div class="reasoning-box">{result["explanation"]}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="disclaimer-box">
          ⚠️ <strong>Clinical Disclaimer:</strong> This AI recommendation is a decision-support
          tool only. Always correlate with local antibiogram data, patient allergies, renal
          function, and clinical judgement. Final treatment decisions remain with the
          treating physician.
        </div>
        """, unsafe_allow_html=True)

        if clinical_notes:
            st.markdown(f"""
            <div style="background:#102030; border-left:4px solid #2980b9; padding:12px 16px;
                        border-radius:0 8px 8px 0; font-size:13px; margin-top:14px;
                        color:#d6e8f5;">
              <strong style="color:#7ec8e3;">Clinical Notes:</strong><br>{clinical_notes}
            </div>
            """, unsafe_allow_html=True)

    # ── resistance profile table ───────────────────────────────────────────────
    with col_table:
        st.markdown("**Antibiotic Resistance Profile**")

        ranked_data = result["ranked"]
        rows = []
        for entry in ranked_data:
            prob = entry["resistance_probability"]
            if prob < 0.30:
                label = "🟢 Likely Susceptible"
                bg    = "#0d3320"
            elif prob < 0.70:
                label = "🟡 Intermediate"
                bg    = "#2a1f00"
            else:
                label = "🔴 Likely Resistant"
                bg    = "#2d0f0f"
            rows.append({
                "Antibiotic":             entry["antibiotic"],
                "Resistance Prob.":       f"{prob:.1%}",
                "Susceptibility":         label,
                "_bg":                    bg,
            })

        df_display = pd.DataFrame(rows)
        df_display.index = range(1, len(df_display) + 1)

        bg_list = df_display["_bg"].tolist()
        df_show = df_display.drop(columns=["_bg"])

        def _row_color(row):
            idx = row.name - 1
            color = bg_list[idx] if idx < len(bg_list) else "#1e2330"
            return [f"background-color: {color}; color: #e8f0f8"] * len(row)

        st.dataframe(
            df_show.style.apply(_row_color, axis=1),
            use_container_width=True,
            height=min(50 + 35 * len(df_show), 520),
        )

        # summary metrics
        n_sus  = sum(1 for r in ranked_data if r["resistance_probability"] <  0.30)
        n_int  = sum(1 for r in ranked_data if 0.30 <= r["resistance_probability"] < 0.70)
        n_res  = sum(1 for r in ranked_data if r["resistance_probability"] >= 0.70)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🟢 Likely Susceptible", n_sus)
        mc2.metric("🟡 Intermediate",       n_int)
        mc3.metric("🔴 Likely Resistant",    n_res)

        # model info badge
        _p = MODEL_PERFORMANCE[selected_model]
        st.caption(
            f"Model: **{selected_model}** · ROC-AUC {_p['ROC-AUC']} · "
            f"PR-AUC {_p['PR-AUC']} · Threshold {OPTIMAL_THRESHOLDS[selected_model]} "
            f"(paper-optimal, Table 4)"
        )

    # ── SHAP explainability panel ─────────────────────────────────────────────
    if result.get("shap_features"):
        st.markdown("---")
        st.markdown("### 🔬 SHAP Feature Importance — What Drove This Prediction?")
        st.markdown(
            f"The following features had the largest influence on the resistance "
            f"prediction for **{top_antibiotic_for_shap}** "
            f"(SHAP values from {selected_model} TreeExplainer, per paper Section 4.3.6)."
        )
        shap_rows = []
        for f in result["shap_features"]:
            shap_rows.append({
                "Feature":    f["feature"],
                "SHAP Value": f"{f['shap_value']:+.4f}",
                "Effect":     f["direction"],
            })
        shap_df = pd.DataFrame(shap_rows)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
        st.caption(
            "Negative SHAP → feature decreases resistance probability (favours susceptibility). "
            "Positive SHAP → feature increases resistance probability."
        )

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div style="margin-bottom:10px; padding:12px 20px; background:#111a28;
              border:1px solid #2a3a52; border-radius:8px; text-align:left;
              font-size:12px; color:#a8c4d8 !important;">
    <span style="font-size:11px; font-weight:700; text-transform:uppercase;
                 letter-spacing:.8px; color:#5dade2;">📄 Citation</span><br><br>
    Almalki&nbsp;Z, Althagafi&nbsp;A, Al-Shareef&nbsp;S.
    <em>Comparative Evaluation of Ensemble Machine Learning Models for Predicting
    Antibacterial Resistance from Electronic Health Records.</em>
    <strong>Discover AI</strong>, Springer Nature (2026).
    DOI: <a href="https://link.springer.com/article/10.1007/s44163-026-01436-4" target="_blank" style="color:#5dade2;">10.1007/s44163-026-01436-4</a><br>
    <span style="font-size:11px; color:#7a90a8;">
      This clinical decision support system is the applied implementation of the above research.
      ML models, thresholds, and performance metrics are derived directly from the paper.
    </span>
  </div>
  AMR Clinical Decision Support System &nbsp;|&nbsp; GROUP-6-UQU-CS6117
  &nbsp;|&nbsp; Umm Al-Qura University &nbsp;|&nbsp; CS6117 — Advanced Topics in AI<br>
  <span style="font-size:11px;">For research and educational use only. Not a substitute for clinical judgement.</span>
</div>
""", unsafe_allow_html=True)
