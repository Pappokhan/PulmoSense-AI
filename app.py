import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
import base64
import os
import pandas as pd

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import cm, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

if platform.system() != 'Windows':
    pio.kaleido.scope.chromium_args += (
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--single-process",
        "--disable-extensions",
        "--disable-features=TranslateUI",
        "--disable-ipc-flooding-protection",
    )


st.set_page_config(
    page_title="PulmoSense AI • Lung Cancer Risk Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

COLORS = {
    'primary': '#0F4C81',
    'secondary': '#2E86AB',
    'accent': '#6A4C93',
    'success': '#2E8B57',
    'warning': '#FF8C42',
    'danger': '#D32F2F',
    'light_bg': '#F8FAFC',
    'dark_text': '#1E293B',
    'border': '#E2E8F0',
    'info': '#3B82F6',
    'card_bg': '#FFFFFF',
    'hover_bg': '#F1F5F9'
}

st.markdown(f"""
<style>
    /* Main container */
    .main .block-container {{
        max-width: 1400px;
        padding: 1.5rem 2rem 1.5rem 2rem;
        margin: 0 auto;
    }}

    /* Header styling */
    .research-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15,76,129,0.15);
        text-align: center;
    }}

    .research-title {{
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}

    .research-subtitle {{
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95);
        margin-top: 0.5rem;
    }}

    .institution-badge {{
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 1.2rem;
        border-radius: 25px;
        display: inline-block;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        backdrop-filter: blur(5px);
    }}

    /* Card styling - Equal height with perfect alignment */
    .academic-card {{
        background: {COLORS['card_bg']};
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid {COLORS['border']};
        margin-bottom: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }}

    .academic-card:hover {{
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }}

    .card-title {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {COLORS['primary']};
        border-bottom: 3px solid {COLORS['secondary']};
        padding-bottom: 0.6rem;
        margin-bottom: 1.3rem;
        display: inline-block;
        width: auto;
    }}

    /* Section headers inside cards */
    .section-header {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {COLORS['secondary']};
        margin: 0.8rem 0 0.8rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px dashed {COLORS['border']};
    }}

    /* Result container */
    .result-container {{
        background: linear-gradient(135deg, white 0%, {COLORS['light_bg']} 100%);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 6px solid {COLORS['accent']};
        margin: 1.5rem 0;
    }}

    .risk-level {{
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin: 0.3rem 0;
    }}

    .probability {{
        font-size: 1.3rem;
        text-align: center;
        margin: 0.3rem 0;
    }}

    .recommendation-box {{
        background: #FFF8F0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 4px solid {COLORS['warning']};
        margin-top: 1rem;
        line-height: 1.5;
    }}

    /* Metrics grid - Perfect 4 column layout */
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin: 1.2rem 0;
    }}

    .metric-card {{
        background: {COLORS['card_bg']};
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid {COLORS['border']};
        transition: all 0.3s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }}

    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        border-color: {COLORS['secondary']};
    }}

    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS['primary']};
        line-height: 1.2;
    }}

    .metric-label {{
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.3rem;
        font-weight: 500;
    }}

    .metric-status {{
        font-size: 0.7rem;
        margin-top: 0.4rem;
        font-weight: 600;
    }}

    /* Two column grid for health metrics */
    .health-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.2rem;
        margin: 1.2rem 0;
    }}

    /* Button styling */
    .stButton {{
        width: 100%;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        border: none;
        transition: all 0.3s;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(15,76,129,0.2);
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15,76,129,0.3);
    }}

    /* Divider */
    .custom-divider {{
        height: 2px;
        background: linear-gradient(to right, transparent, {COLORS['secondary']}50, transparent);
        margin: 1.5rem 0;
    }}

    /* Footer */
    .research-footer {{
        text-align: center;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid {COLORS['border']};
        color: #64748B;
        font-size: 0.85rem;
    }}

    /* Citation box */
    .citation-box {{
        background: {COLORS['light_bg']};
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-size: 0.75rem;
        margin-top: 1rem;
        border: 1px solid {COLORS['border']};
        text-align: center;
    }}

    /* Status badges */
    .status-badge {{
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }}

    .status-low {{
        background: {COLORS['success']}20;
        color: {COLORS['success']};
    }}

    .status-moderate {{
        background: {COLORS['warning']}20;
        color: {COLORS['warning']};
    }}

    .status-high {{
        background: {COLORS['danger']}20;
        color: {COLORS['danger']};
    }}

    /* Alert box */
    .clinical-alert {{
        background: {COLORS['danger']}10;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['danger']};
        margin: 1rem 0;
        font-size: 0.85rem;
    }}

    .warning-alert {{
        background: {COLORS['warning']}10;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['warning']};
        margin: 1rem 0;
        font-size: 0.85rem;
    }}

    /* Chart container */
    .chart-container {{
        background: {COLORS['card_bg']};
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid {COLORS['border']};
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}

    /* Input field styling */
    .stSelectbox div[data-baseweb="select"] {{
        border-radius: 8px;
    }}

    .stSlider div[data-baseweb="slider"] {{
        margin-top: 0.5rem;
    }}

    /* Label styling */
    .stSelectbox label, .stSlider label {{
        font-weight: 500;
        color: {COLORS['dark_text']};
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }}

    /* Caption styling */
    .stCaption {{
        font-size: 0.7rem;
        margin-top: -0.2rem;
        margin-bottom: 0.5rem;
    }}

    /* Responsive */
    @media (max-width: 1200px) {{
        .metrics-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}

    @media (max-width: 768px) {{
        .metrics-grid {{
            grid-template-columns: 1fr;
        }}
        .health-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['light_bg']};
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['secondary']};
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['primary']};
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = 'LungCancer_Stacking_100.pkl'
    if not os.path.exists(model_path):
        st.error(f"""
        ⚠️ **Model File Not Found**

        The required model file `{model_path}` is missing. Please ensure it's in the correct directory.
        """)
        st.stop()

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


model = load_model()

# Header Section
st.markdown(f"""
<div class="research-header">
    <div class="research-title">🫁 PulmoSense AI</div>
    <div class="research-subtitle">Lung Cancer Risk Prediction System | Clinical Decision Support Tool</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large", vertical_alignment="top")

with col_left:
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Clinical Assessment Form</div>', unsafe_allow_html=True)

    col_l1, col_l2 = st.columns(2, gap="medium")

    with col_l1:
        st.markdown('<div class="section-header">🚬 Lifestyle Factors</div>', unsafe_allow_html=True)
        smoking = st.selectbox(
            "Smoking Status",
            ["Never", "Former", "Current"],
            index=2,
            help="Current: Active smoker | Former: Quit >1 year"
        )
        breathing_issue = st.selectbox("Breathing Difficulty", ["No", "Yes"], index=1)
        throat_discomfort = st.selectbox("Throat Irritation", ["No", "Yes"], index=1)
        pollution = st.selectbox("Pollution Exposure", ["No", "Yes"], index=1)

    with col_l2:
        st.markdown('<div class="section-header">🧬 Genetic Factors</div>', unsafe_allow_html=True)
        family_cancer = st.selectbox("Family Cancer History", ["No", "Yes"], index=0)
        family_smoking = st.selectbox("Family Smoking History", ["No", "Yes"], index=0)

        st.markdown('<div class="section-header" style="margin-top: 0.5rem;">📊 Clinical Metrics</div>',
                    unsafe_allow_html=True)
        age = st.slider("Age (years)", 18, 100, 68, key="age_slider")
        spo2 = st.slider("SpO₂ (%)", 85, 100, 91, key="spo2_slider")

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="academic-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💪 Health Status Assessment</div>', unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2, gap="medium")

    with col_r1:
        st.markdown('<div class="section-header">⚡ Vital Signs</div>', unsafe_allow_html=True)
        energy = st.slider(
            "Energy Level",
            1, 10, 4,
            help="1=Severe fatigue → 10=High energy"
        )
        if energy < 5:
            st.caption("⚠️ Low energy level")
        else:
            st.caption("✓ Normal energy level")

    with col_r2:
        st.markdown('<div class="section-header">🛡️ Immune Status</div>', unsafe_allow_html=True)
        immunity = st.slider(
            "Immune Health",
            1, 10, 3,
            help="1=Immunocompromised → 10=Excellent"
        )
        if immunity < 6:
            st.caption("⚠️ Weakened immunity")
        else:
            st.caption("✓ Good immunity")

    if spo2 < 90:
        st.markdown(f"""
        <div class="clinical-alert">
            ⚠️ <strong>Critical Alert:</strong> Hypoxemia detected (SpO₂ = {spo2}%).<br>
            Immediate clinical evaluation required.
        </div>
        """, unsafe_allow_html=True)
    elif spo2 < 94:
        st.markdown(f"""
        <div class="warning-alert">
            ⚠️ <strong>Clinical Note:</strong> Below normal SpO₂ ({spo2}%).<br>
            Monitor oxygen saturation closely.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Button Section
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_clicked = st.button("🔬 Analyze Risk Profile", type="primary", use_container_width=True)

# Results Section
if analyze_clicked:
    with st.spinner("🧠 Processing clinical data with stacking ensemble model..."):

        # Prepare features
        features = np.array([[
            ["Never", "Former", "Current"].index(smoking),
            ["No", "Yes"].index(breathing_issue),
            ["No", "Yes"].index(throat_discomfort),
            ["No", "Yes"].index(family_smoking),
            energy,
            immunity,
            spo2,
            ["No", "Yes"].index(family_cancer),
            ["No", "Yes"].index(pollution),
            age
        ]])

        # Get prediction
        prob = model.predict_proba(features)[0][1]
        risk_pct = round(prob * 100, 1)

        # Risk classification
        if prob < 0.3:
            level, color, advice, risk_class = "Low Risk", COLORS[
                'success'], "Continue healthy habits. Annual chest X-ray recommended for patients >50 years or with risk factors. Maintain regular follow-ups with primary care physician.", "Low"
        elif prob < 0.7:
            level, color, advice, risk_class = "Moderate Risk", COLORS[
                'warning'], "Low-dose CT scan (LDCT) recommended within 3 months. Pulmonology consultation advised. Consider smoking cessation program if applicable. Monitor respiratory symptoms closely.", "Moderate"
        else:
            level, color, advice, risk_class = "High Risk", COLORS[
                'danger'], "URGENT: Immediate referral to oncology. LDCT within 48 hours. Comprehensive pulmonary function tests required. Consider biopsy. Emergency assessment recommended for symptomatic patients.", "High"

        # Results Container
        st.markdown(f"""
        <div class="result-container">
            <div class="risk-level" style="color:{color}">
                🎯 {level}
                <span class="status-badge status-{risk_class.lower()}">{risk_class}</span>
            </div>
            <div class="probability">
                Predicted Probability: <strong style="color:{color}; font-size:2rem;">{prob:.1%}</strong>
            </div>
            <div class="recommendation-box">
                <strong>📋 Clinical Recommendation:</strong><br>
                {advice}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chart Section - Title completely removed
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            title=None,  # Completely remove the title
            delta={'reference': 50, 'increasing': {'color': COLORS['danger']},
                   'decreasing': {'color': COLORS['success']}},
            number={'suffix': "%", 'font': {'size': 60, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': COLORS['dark_text'], 'tickfont': {'size': 12}},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "#FFFFFF",
                'borderwidth': 2,
                'bordercolor': COLORS['border'],
                'steps': [
                    {'range': [0, 30], 'color': "#DCFCE7"},
                    {'range': [30, 70], 'color': "#FEF9C3"},
                    {'range': [70, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {
                    'line': {'color': COLORS['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            height=400,
            margin=dict(t=60, b=40, l=40, r=40),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['dark_text'], 'family': 'Arial', 'size': 12}
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Detailed Factor Analysis - 4 Column Grid
        st.markdown('<div class="academic-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Detailed Factor Analysis</div>', unsafe_allow_html=True)

        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)

        # Age Analysis
        age_risk = "High" if age > 60 else "Moderate" if age > 50 else "Low"
        age_color = COLORS['danger'] if age > 60 else COLORS['warning'] if age > 50 else COLORS['success']
        age_icon = "🔴" if age > 60 else "🟡" if age > 50 else "🟢"

        # Smoking Analysis
        smoking_risk = "Critical" if smoking == "Current" else "Reduced" if smoking == "Former" else "Minimal"
        smoking_color = COLORS['danger'] if smoking == "Current" else COLORS['warning'] if smoking == "Former" else \
        COLORS['success']
        smoking_icon = "🚬" if smoking == "Current" else "✅" if smoking == "Former" else "✓"

        # SpO2 Analysis
        spo2_status = "Critical" if spo2 < 90 else "Low" if spo2 < 94 else "Normal"
        spo2_color = COLORS['danger'] if spo2 < 90 else COLORS['warning'] if spo2 < 94 else COLORS['success']
        spo2_icon = "⚠️" if spo2 < 90 else "📊" if spo2 < 94 else "✓"

        # Family History Analysis
        family_status = "Present" if family_cancer == "Yes" else "Absent"
        family_color = COLORS['warning'] if family_cancer == "Yes" else COLORS['success']
        family_icon = "🧬" if family_cancer == "Yes" else "✓"

        col1, col2, col3, col4 = st.columns(4, gap="small")

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{age_icon} {age}</div>
                <div class="metric-label">Age (years)</div>
                <div class="metric-status" style="color:{age_color}">● {age_risk} Risk</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{smoking_icon} {smoking}</div>
                <div class="metric-label">Smoking Status</div>
                <div class="metric-status" style="color:{smoking_color}">● {smoking_risk}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{spo2_icon} {spo2}%</div>
                <div class="metric-label">SpO₂ Level</div>
                <div class="metric-status" style="color:{spo2_color}">● {spo2_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{family_icon} {family_status}</div>
                <div class="metric-label">Family History</div>
                <div class="metric-status" style="color:{family_color}">● {'Genetic factor' if family_cancer == 'Yes' else 'No known risk'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Health Status Analysis - 2 Column Grid
        st.markdown('<div class="academic-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💪 Health Status Analysis</div>', unsafe_allow_html=True)

        st.markdown('<div class="health-grid">', unsafe_allow_html=True)

        col_h1, col_h2 = st.columns(2, gap="medium")

        with col_h1:
            energy_risk = "Compromised" if energy < 5 else "Normal" if energy < 8 else "Excellent"
            energy_color = COLORS['danger'] if energy < 5 else COLORS['warning'] if energy < 8 else COLORS['success']
            energy_icon = "⚡" if energy < 5 else "💪" if energy < 8 else "🌟"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{energy_icon} {energy}/10</div>
                <div class="metric-label">Energy Level</div>
                <div class="metric-status" style="color:{energy_color}">● {energy_risk}</div>
                <div style="margin-top:0.6rem; font-size:0.7rem; color:#64748B;">
                    {'Severe fatigue detected' if energy < 5 else 'Normal energy levels' if energy < 8 else 'High vitality'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_h2:
            immune_risk = "Immunocompromised" if immunity < 5 else "Moderate" if immunity < 8 else "Strong"
            immune_color = COLORS['danger'] if immunity < 5 else COLORS['warning'] if immunity < 8 else COLORS[
                'success']
            immune_icon = "🛡️" if immunity < 5 else "💪" if immunity < 8 else "🏆"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{immune_icon} {immunity}/10</div>
                <div class="metric-label">Immune Health</div>
                <div class="metric-status" style="color:{immune_color}">● {immune_risk}</div>
                <div style="margin-top:0.6rem; font-size:0.7rem; color:#64748B;">
                    {'Weakened immune response' if immunity < 5 else 'Adequate immunity' if immunity < 8 else 'Robust immune system'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Store results for PDF
        st.session_state.result = {
            "prob": prob,
            "level": level,
            "color": color,
            "advice": advice,
            "fig": fig,
            "risk_pct": risk_pct,
            "date": datetime.now(),
            "inputs": {
                "Age": age,
                "Smoking Status": smoking,
                "Breathing Difficulty": breathing_issue,
                "Throat Irritation": throat_discomfort,
                "Pollution Exposure": pollution,
                "Family Cancer History": family_cancer,
                "Family Smoking History": family_smoking,
                "Energy Level": f"{energy}/10",
                "Immune Health": f"{immunity}/10",
                "SpO₂ Level": f"{spo2}%"
            }
        }


# PDF GENERATION
def create_research_pdf():
    buffer = BytesIO()

    try:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(A4),
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm
        )

        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            name="Title", fontSize=22, textColor=colors.HexColor(COLORS['primary']),
            spaceAfter=10, alignment=1, fontName='Helvetica-Bold'
        )
        subtitle_style = ParagraphStyle(
            name="Subtitle", fontSize=11, textColor=colors.HexColor(COLORS['secondary']),
            alignment=1, spaceAfter=15, fontName='Helvetica'
        )
        section_style = ParagraphStyle(
            name="Section", fontSize=13, textColor=colors.HexColor(COLORS['primary']),
            spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold'
        )

        # Header
        story.append(Paragraph("PulmoSense AI", title_style))
        story.append(Paragraph("Lung Cancer Risk Assessment Report", subtitle_style))
        story.append(Paragraph(f"Report ID: ALA-{datetime.now().strftime('%Y%m%d%H%M%S')}", styles["Normal"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles["Normal"]))
        story.append(Spacer(1, 0.6 * cm))

        # Patient Data
        story.append(Paragraph("Clinical Risk Factors", section_style))

        data_items = list(st.session_state.result["inputs"].items())
        data = [["Parameter", "Value"]]
        for k, v in data_items:
            data.append([k, str(v)])

        table = Table(data, colWidths=[7 * cm, 6 * cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLORS['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(COLORS['light_bg'])),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.6 * cm))

        # Results
        story.append(Paragraph("Risk Assessment Results", section_style))
        story.append(Paragraph(
            f"<b>Risk Classification:</b> <font color='{st.session_state.result['color']}'><b>{st.session_state.result['level']}</b></font>",
            styles["Normal"]))
        story.append(
            Paragraph(f"<b>Probability Score:</b> <b>{st.session_state.result['prob']:.1%}</b>", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("<b>Clinical Recommendation:</b>", styles["Normal"]))
        story.append(Paragraph(st.session_state.result["advice"],
                               ParagraphStyle(name="Advice", fontSize=10, leading=14, spaceAfter=15)))

        # Disclaimer
        story.append(Paragraph(
            "<b>Disclaimer:</b> For research and educational purposes only. Not a substitute for professional medical advice.",
            ParagraphStyle(name="Disc", fontSize=8, textColor=colors.gray, alignment=1)))

        doc.build(story)

    except Exception as e:
        return None

    return buffer.getvalue()


# Download Button
if "result" in st.session_state:
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    pdf_data = create_research_pdf()

    if pdf_data:
        b64 = base64.b64encode(pdf_data).decode()
        filename = f"PulmoSense_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.markdown(f'''
            <div style="text-align: center; margin: 0.5rem 0;">
                <a href="data:application/pdf;base64,{b64}" download="{filename}" style="text-decoration: none;">
                    <button style="background: linear-gradient(135deg, {COLORS['accent']}, {COLORS['primary']}); 
                            color: white; padding: 0.8rem 2rem; border: none; border-radius: 12px; 
                            font-size: 1rem; cursor: pointer; font-weight: 600; width: 100%;
                            box-shadow: 0 2px 8px rgba(106,76,147,0.3);">
                        📄 Download Research Report (PDF)
                    </button>
                </a>
            </div>
            ''', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="research-footer">
    <strong>PulmoSense AI</strong> | Research Publication Ready<br>
    <small>© 2026 • PulmoSense AI Lung Cancer Risk Prediction System.</em>, 2026</small>
</div>
""", unsafe_allow_html=True)
