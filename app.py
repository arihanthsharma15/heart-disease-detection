"""
Heart Disease Detection — Interactive Clinical Risk Assessment Application

Streamlined, user-friendly interface designed for clear clinical demonstrations:
1. Patient Clinical Input Form (with input bounds & help guides)
2. One-click Presentation Demo Cases (Low, Moderate, High Risk)
3. 3-Tier Clinical Risk Stratification (Low <55%, Moderate 55-75%, High >75%)
4. Multi-Model Consensus (Random Forest, SVM, Weighted Hybrid)
"""

import sys
from pathlib import Path

# Ensure src/ is importable
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st
import pandas as pd
import numpy as np

from heart_disease.config import (
    ALL_FEATURES,
    DECISION_THRESHOLD,
    DISCLAIMER,
    RISK_TIER_MODERATE_MIN,
    RISK_TIER_HIGH_MIN,
)
from heart_disease.predict import predict_from_dict
from heart_disease.schemas import PatientFeatures, classify_risk_tier


# --- Page Configuration ---
st.set_page_config(
    page_title="Cardiovascular Disease Risk Detector",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Clean Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #e63946;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #457b9d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.3rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-left: 6px solid #1d3557;
        margin-bottom: 1.2rem;
    }
    .risk-high {
        border-left: 6px solid #e63946 !important;
        background-color: #fff5f5 !important;
    }
    .risk-moderate {
        border-left: 6px solid #f4a261 !important;
        background-color: #fffdf5 !important;
    }
    .risk-low {
        border-left: 6px solid #2a9d8f !important;
        background-color: #f0fdf4 !important;
    }
    .tier-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: white;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 0.9rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #5d4037;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 10 Pre-Tested Demo Cases for College Presentations ---
DEMO_CASES = {
    "Case 1: Young Healthy Athlete (Expected: Low Risk ~4%)": {
        "age": 35, "sex": 0, "cp": 2, "trestbps": 115, "chol": 172,
        "fbs": 0, "restecg": 0, "thalach": 178, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    },
    "Case 2: Normal Middle-Aged Adult (Expected: Low Risk ~7%)": {
        "age": 45, "sex": 1, "cp": 3, "trestbps": 120, "chol": 210,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 0.4, "slope": 1, "ca": 0, "thal": 3
    },
    "Case 3: Healthy Active Senior (Expected: Low Risk ~8%)": {
        "age": 65, "sex": 0, "cp": 3, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 0, "thalach": 155, "exang": 0,
        "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3
    },
    "Case 4: Borderline / Low-to-Moderate (Expected: Low Risk ~12%)": {
        "age": 52, "sex": 1, "cp": 3, "trestbps": 132, "chol": 225,
        "fbs": 0, "restecg": 1, "thalach": 145, "exang": 0,
        "oldpeak": 0.8, "slope": 1, "ca": 0, "thal": 3
    },
    "Case 5: Intermediate / Moderate Risk (Expected: Moderate ~65%)": {
        "age": 53, "sex": 1, "cp": 4, "trestbps": 130, "chol": 246,
        "fbs": 0, "restecg": 0, "thalach": 140, "exang": 0,
        "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 6
    },
    "Case 6: Early Ischemia Signs (Expected: High Risk ~83%)": {
        "age": 54, "sex": 1, "cp": 4, "trestbps": 135, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 130, "exang": 1,
        "oldpeak": 1.6, "slope": 2, "ca": 1, "thal": 3
    },
    "Case 7: High Risk (1 Vessel Blocked) (Expected: High Risk ~99%)": {
        "age": 60, "sex": 1, "cp": 4, "trestbps": 145, "chol": 282,
        "fbs": 1, "restecg": 2, "thalach": 122, "exang": 1,
        "oldpeak": 2.2, "slope": 2, "ca": 1, "thal": 7
    },
    "Case 8: Severe Multi-Vessel CAD (Expected: High Risk ~99%)": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270,
        "fbs": 0, "restecg": 2, "thalach": 115, "exang": 1,
        "oldpeak": 2.8, "slope": 2, "ca": 2, "thal": 7
    },
    "Case 9: Senior Female with Angina (Expected: High Risk ~95%)": {
        "age": 67, "sex": 0, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
        "oldpeak": 3.2, "slope": 2, "ca": 2, "thal": 7
    },
    "Case 10: Critical Tri-Vessel CAD (Expected: High Risk ~94%)": {
        "age": 68, "sex": 1, "cp": 4, "trestbps": 170, "chol": 315,
        "fbs": 1, "restecg": 2, "thalach": 98, "exang": 1,
        "oldpeak": 3.8, "slope": 3, "ca": 3, "thal": 7
    }
}

PRESET_HEALTHY = DEMO_CASES["Case 1: Young Healthy Athlete (Expected: Low Risk ~4%)"]
PRESET_MODERATE = DEMO_CASES["Case 5: Intermediate / Moderate Risk (Expected: Moderate ~65%)"]
PRESET_HIGH_RISK = DEMO_CASES["Case 8: Severe Multi-Vessel CAD (Expected: High Risk ~99%)"]


def load_preset(preset_data):
    for k, v in preset_data.items():
        st.session_state[f"input_{k}"] = v


# --- Left Sidebar: Demo Case Loader & Information ---
st.sidebar.image("https://img.icons8.com/fluency/96/heart-health.png", width=64)
st.sidebar.title("CardioRisk AI")
st.sidebar.markdown("**Cardiovascular Disease Risk Detector**")
st.sidebar.caption("Hybrid Model: Random Forest (60%) + SVM (40%)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Demo Patient Cases")
st.sidebar.caption("Pick any case to auto-fill the form:")

selected_case_name = st.sidebar.selectbox(
    "Select Patient Sample:",
    options=list(DEMO_CASES.keys()),
    index=0
)
if st.sidebar.button("👉 Load Case into Form", use_container_width=True, type="secondary"):
    load_preset(DEMO_CASES[selected_case_name])
    st.rerun()

st.sidebar.markdown("##### Quick 1-Click Presets")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)
with col_p1:
    if st.button("🟢 Low", use_container_width=True, help="Load Healthy (<55%) profile"):
        load_preset(PRESET_HEALTHY)
        st.rerun()
with col_p2:
    if st.button("🟡 Mod", use_container_width=True, help="Load Moderate (55-75%) profile"):
        load_preset(PRESET_MODERATE)
        st.rerun()
with col_p3:
    if st.button("🔴 High", use_container_width=True, help="Load High (>75%) profile"):
        load_preset(PRESET_HIGH_RISK)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "🩺 **Risk Stratification Scale:**\n\n"
    "• 🟢 **Low Risk:** $< 55\\%$\n"
    "  *(Routine preventive care)*\n\n"
    "• 🟡 **Moderate Risk:** $55\\% - 75\\%$\n"
    "  *(Borderline / Non-invasive tests)*\n\n"
    "• 🔴 **High Risk:** $> 75\\%$\n"
    "  *(Cardiologist consult / Angiogram)*"
)


# --- Main Application: Patient Clinical Input Form ---
st.markdown('<div class="main-header">Cardiovascular Disease Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter patient vitals and test measurements below to calculate cardiovascular disease risk.</div>', unsafe_allow_html=True)

with st.form("patient_form"):
    st.markdown("#### 1. Patient Demographics & Vitals")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120,
            value=st.session_state.get("input_age", 55),
            help="Patient age in years (1 to 120)"
        )
    with col2:
        sex_choice = st.selectbox(
            "Biological Sex",
            options=[("Male (1)", 1), ("Female (0)", 0)],
            index=0 if st.session_state.get("input_sex", 1) == 1 else 1,
            format_func=lambda x: x[0]
        )
        sex = sex_choice[1]
    with col3:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=50, max_value=300,
            value=st.session_state.get("input_trestbps", 130),
            help="Resting blood pressure measured on admission (mm Hg)"
        )
    with col4:
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)", min_value=50, max_value=700,
            value=st.session_state.get("input_chol", 240),
            help="Serum total cholesterol in mg/dl"
        )

    st.markdown("#### 2. Symptoms & Laboratory Findings")
    col5, col6, col7 = st.columns(3)
    with col5:
        cp_choice = st.selectbox(
            "Chest Pain Type (CP)",
            options=[
                ("Typical Angina (1)", 1),
                ("Atypical Angina (2)", 2),
                ("Non-Anginal Pain (3)", 3),
                ("Asymptomatic (4)", 4)
            ],
            index=st.session_state.get("input_cp", 2) - 1,
            format_func=lambda x: x[0],
            help="Clinical presentation of chest discomfort"
        )
        cp = cp_choice[1]
    with col6:
        fbs_choice = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[("False / Normal (0)", 0), ("True / Elevated (1)", 1)],
            index=1 if st.session_state.get("input_fbs", 0) == 1 else 0,
            format_func=lambda x: x[0],
            help="Fasting blood sugar indicator (diabetes risk)"
        )
        fbs = fbs_choice[1]
    with col7:
        restecg_choice = st.selectbox(
            "Resting ECG Results",
            options=[
                ("Normal (0)", 0),
                ("ST-T Wave Abnormality (1)", 1),
                ("Left Ventricular Hypertrophy (2)", 2)
            ],
            index=st.session_state.get("input_restecg", 0),
            format_func=lambda x: x[0],
            help="Resting electrocardiogram findings"
        )
        restecg = restecg_choice[1]

    st.markdown("#### 3. Exercise Stress Test Findings")
    col8, col9, col10, col11 = st.columns(4)
    with col8:
        thalach = st.number_input(
            "Maximum Heart Rate Achieved (bpm)", min_value=40, max_value=280,
            value=st.session_state.get("input_thalach", 150),
            help="Peak pulse reached during treadmill exercise test"
        )
    with col9:
        exang_choice = st.selectbox(
            "Exercise-Induced Angina",
            options=[("No (0)", 0), ("Yes (1)", 1)],
            index=1 if st.session_state.get("input_exang", 0) == 1 else 0,
            format_func=lambda x: x[0],
            help="Chest pain elicited specifically by physical exertion"
        )
        exang = exang_choice[1]
    with col10:
        oldpeak = st.number_input(
            "ST Depression ('oldpeak')", min_value=-5.0, max_value=10.0, step=0.1,
            value=float(st.session_state.get("input_oldpeak", 1.5)),
            help="ST segment depression on ECG during exercise relative to rest (in mm)"
        )
    with col11:
        slope_choice = st.selectbox(
            "Slope of Peak ST Segment",
            options=[
                ("Upsloping (1)", 1),
                ("Flat (2)", 2),
                ("Downsloping (3)", 3)
            ],
            index=st.session_state.get("input_slope", 2) - 1,
            format_func=lambda x: x[0],
            help="Shape of the ECG ST slope at peak exertion"
        )
        slope = slope_choice[1]

    st.markdown("#### 4. Diagnostic Imaging & Thallium Scan")
    col12, col13 = st.columns(2)
    with col12:
        ca_choice = st.selectbox(
            "Major Vessels Colored by Fluoroscopy (0-3)",
            options=[("0 Vessels (0)", 0), ("1 Vessel (1)", 1), ("2 Vessels (2)", 2), ("3 Vessels (3)", 3)],
            index=st.session_state.get("input_ca", 0),
            format_func=lambda x: x[0],
            help="Number of major coronary blood vessels visualized with dye narrowing"
        )
        ca = ca_choice[1]
    with col13:
        thal_options = [
            ("Normal (3)", 3),
            ("Fixed Defect (6)", 6),
            ("Reversable Defect (7)", 7)
        ]
        current_thal = st.session_state.get("input_thal", 3)
        thal_idx = 0
        if current_thal == 6:
            thal_idx = 1
        elif current_thal == 7:
            thal_idx = 2
        thal_choice = st.selectbox(
            "Thallium Scintigraphy Result",
            options=thal_options,
            index=thal_idx,
            format_func=lambda x: x[0],
            help="Perfusion pattern during cardiac nuclear stress scan"
        )
        thal = thal_choice[1]

    submitted = st.form_submit_button("⚡ Analyze Patient Risk", use_container_width=True, type="primary")


# --- Results Display ---
if submitted:
    patient_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    try:
        prediction_result = predict_from_dict(patient_data)
        res_dict = prediction_result.to_dict()

        hybrid = res_dict["hybrid"]
        rf = res_dict["random_forest"]
        svm = res_dict["svm"]
        weight = res_dict["fusion_weight"]

        p_hybrid = hybrid["probability"]
        pred_hybrid = hybrid["prediction"]
        is_high_risk = pred_hybrid == 1

        st.markdown("---")
        st.markdown("### 📋 Patient Risk Assessment Results")

        # 3-Tier Stratification Analysis
        tier_info = classify_risk_tier(p_hybrid)
        res_class = tier_info["class_name"]
        risk_label = tier_info["label"]
        risk_color = tier_info["color"]
        badge_text = tier_info["badge"]
        recommendation = tier_info["recommendation"]

        # 1. Main Consensus Card
        st.markdown(f"""
        <div class="metric-card {res_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-size: 0.88rem; text-transform: uppercase; color: #555; font-weight: 700; letter-spacing: 0.5px;">Consensus Diagnosis</span>
                <span class="tier-badge" style="background-color: {risk_color};">{badge_text}</span>
            </div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {risk_color}; margin-top: 0.2rem; line-height: 1.2;">
                {risk_label}
            </div>
            <div style="font-size: 1.15rem; margin-top: 0.5rem; color: #222;">
                Overall Estimated Risk: <strong>{p_hybrid * 100:.1f}%</strong> 
                <span style="color: #666; font-size: 0.95rem;">(Binary Classification: <strong>{'Disease Present (1)' if is_high_risk else 'No Disease (0)'}</strong>)</span>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.93rem; color: #333; background: rgba(0,0,0,0.035); padding: 0.75rem; border-radius: 6px; border-left: 3px solid {risk_color};">
                🩺 <strong>Clinical Action Plan:</strong> {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Progress Bar with Risk Zones
        st.progress(float(p_hybrid))
        st.markdown("""
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-top: -0.2rem; margin-bottom: 1.4rem; padding: 0 4px;">
            <span style="color: #2a9d8f; font-weight: 700;">🟢 Low Risk (&lt;55%)</span>
            <span style="color: #f4a261; font-weight: 700;">🟡 Moderate Risk (55% - 75%)</span>
            <span style="color: #e63946; font-weight: 700;">🔴 High Risk (&gt;75%)</span>
        </div>
        """, unsafe_allow_html=True)

        # 3. Model Breakdown Columns
        st.markdown("#### Individual Model Breakdown")
        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            rf_prob = rf["probability"]
            rf_pred_text = "Disease (1)" if rf["prediction"] == 1 else "No Disease (0)"
            st.metric(
                label="🌲 Random Forest Model",
                value=f"{rf_prob * 100:.1f}%",
                delta=f"Output: {rf_pred_text}"
            )
            st.caption(f"Decision Threshold: {rf['threshold']}")

        with col_m2:
            svm_prob = svm["probability"]
            svm_pred_text = "Disease (1)" if svm["prediction"] == 1 else "No Disease (0)"
            st.metric(
                label="⚡ Support Vector Machine",
                value=f"{svm_prob * 100:.1f}%",
                delta=f"Output: {svm_pred_text}"
            )
            st.caption(f"Decision Threshold: {svm['threshold']}")

        with col_m3:
            st.metric(
                label="⚖️ Weighted Hybrid Consensus",
                value=f"{p_hybrid * 100:.1f}%",
                delta=f"Output: {'Disease (1)' if is_high_risk else 'No Disease (0)'}"
            )
            st.caption(f"Ensemble Blend: 60% RF + 40% SVM")

        # 4. Simple Clinical Disclaimer
        st.markdown(f"""
        <div class="disclaimer-box">
            ⚠️ <strong>Medical Research Disclaimer:</strong> {hybrid['disclaimer']}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Inference Error: {e}")
