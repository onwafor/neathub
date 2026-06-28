# GRID HOSTING CAPACITY ASSESSMENT DASHBOARD

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Grid Hosting Capacity Assessment Tool",
    layout="wide"
)

st.markdown("""
<style>

.notice-bar{
    width:100%;
    background:#003049;
    border-left:5px solid #00e5ff;
    border-right:5px solid #00e5ff;
    overflow:hidden;
    padding:10px 0;
    margin-bottom:15px;
}

.notice-bar span{
    display:inline-block;
    white-space:nowrap;
    color:white;
    font-size:16px;
    font-weight:600;
    animation:scroll-left 38s linear infinite;
    padding-left:100%;
}

@keyframes scroll-left{
    from{
        transform:translateX(0);
    }
    to{
        transform:translateX(-100%);
    }
}

</style>

<div class="notice-bar">
<span>

⚠️ PROTOTYPE VERSION • This is the Prototype Edition of the Grid Hosting Capacity (GHC) Intelligence Engine...  For the full licensed institutional deployment and customization, please contact the National Energy AI Training Hub (NEAT-Hub), Department of Mechanical Engineering, Ahmadu Bello University, Zaria, or visit www . neat-hub . abu . edu . ng/contact.

</span>
</div>

""", unsafe_allow_html=True)


plt.style.use('dark_background')

# =====================================================
# CONTROL ROOM CSS THEME
# =====================================================
st.markdown(
    """
    <style>

    header[data-testid="stHeader"] {
    display: none;
    }

    .block-container {
    padding-top: 0rem;
    }

    [data-testid="stSidebar"] {
    min-width: 650px;
    max-width: 750px;
    }

    /* Hide hamburger menu */
    #MainMenu {
        visibility: hidden;
    }

    /* Hide footer */
    footer {
        visibility: hidden;
    }

    /* Hide header */
    header {
        visibility: hidden;
    }

    /* Hide top-right toolbar */
    [data-testid="stToolbar"] {
        display: none;
    }

    /* Hide deploy button */
    [data-testid="stDecoration"] {
        display: none;
    }

    



    .stApp {
        background-color: #081018;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }

    p, div {
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1b2a;
    }

    /* =====================================================
       FIX DROPDOWN / SELECTBOX TEXT VISIBILITY
    ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #102030 !important;
        color: white !important;
    }

    div[data-baseweb="select"] span {
        color: white !important;
    }

    div[role="listbox"] {
        background-color: #102030 !important;
        color: white !important;
    }

    /* DROPDOWN MENU ITEMS */
    ul {
        background-color: #102030 !important;
    }

    li {
        background-color: #102030 !important;
        color: white !important;
    }

    li:hover {
        background-color: #1b3a4b !important;
        color: white !important;
    }

    div[role="option"] {
        background-color: #102030 !important;
        color: white !important;
    }

    div[role="option"] * {
        color: white !important;
    }

    div[role="option"]:hover {
        background-color: #1b3a4b !important;
        color: white !important;
    }

    /* FORCE VISIBILITY FOR STREAMLIT SELECTBOX OPTIONS */
    .stSelectbox div[data-baseweb="select"] ul {
        background-color: #102030 !important;
    }

    .stSelectbox div[data-baseweb="select"] li {
        color: white !important;
        background-color: #102030 !important;
    }

    .stSelectbox div[data-baseweb="select"] li:hover {
        background-color: #1b3a4b !important;
        color: white !important;
    }

    label {
        color: white !important;
        font-weight: bold;
    }

    .metric-card {
        background: linear-gradient(145deg, #102030, #0b1722);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 0 15px rgba(0,255,255,0.08);
        text-align: center;
        margin-bottom: 15px;
    }

    .metric-title {
        font-size: 16px;
        color: #9ecfff;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #102030, #0b1722);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 0 10px rgba(0,255,255,0.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL & ENCODERS
# =====================================================
@st.cache_resource

def load_model():
    with open(BASE_DIR / "ghc_model_clean.pkl", "rb") as f:
    #with open("ghc_model_clean.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoders_clean.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, encoders = load_model()

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data

def load_data():
    #return pd.read_csv("ghc.csv")
    return pd.read_csv(BASE_DIR / "ghc.csv")

df = load_data()

# =====================================================
# TITLE
# =====================================================
colt1, colt2, colt3 = st.columns([1,4,2])

with colt1:
    #st.image("NEAT_Hub_LOGO.png")
    st.image(str(BASE_DIR / "NEAT_Hub_LOGO.png"))
with colt2:
    st.markdown(
        "<h1 style='text-align: left; font-size: 30px;color: grey ;font-weight: bold;'>"
        " Grid Hosting Capacity Intelligence Engine</h1>",
        unsafe_allow_html=True
    )

with colt3:
    st.markdown(
        "<h1 style='text-align: left; font-size: 15px;color: cyan ;font-style: italic;font-weight: regular;'>"
        "--- See Beyond the Cables</h1>",
        unsafe_allow_html=True
    )

st.divider()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("<p style='text-align: justify; color: white ; font-weight: bold; font-size: 18px;'> This platform evaluates whether an electricity distribution network can safely host renewable energy injection under different operational and environmental conditions "
" .</p>", unsafe_allow_html=True )

#st.sidebar.image("logo.png")
st.sidebar.image(str(BASE_DIR / "logo.png"))
st.sidebar.markdown("# Scenario Configuration")
feeder_options = sorted(df["feeder_id"].unique())
selected_feeder = st.sidebar.selectbox("Select Feeder", feeder_options)

feeder_df = df[df["feeder_id"] == selected_feeder].copy()

if feeder_df.empty:
    st.error("No feeder data available.")
    st.stop()

feeder_df["month_dt"] = pd.to_datetime(feeder_df["month"])
baseline = feeder_df.sort_values("month_dt", ascending=False).iloc[0]

# =====================================================
# INPUTS
# =====================================================
proposed_dg = st.sidebar.slider(
    "Proposed DG (kW)",
    0,
    5000,
    int(baseline["proposed_dg_kw"])
)

dg_type = st.sidebar.selectbox(
    "DG Type",
    ["Solar", "Wind", "Hybrid"]
)

coincidence_factor = st.sidebar.slider(
    "Coincidence Factor",
    0.0,
    1.0,
    float(baseline["coincidence_factor"])
)

season = st.sidebar.selectbox(
    "Season",
    ["Dry", "Rainy"]
)

solar_irradiance_index = st.sidebar.slider(
    "Solar Irradiance Index",
    0.0,
    1.0,
    float(baseline["solar_irradiance_index"])
)

peak_load_kw = st.sidebar.slider(
    "Peak Load (kW)",
    100,
    10000,
    int(baseline["peak_load_kw"])
)

average_load_kw = st.sidebar.slider(
    "Average Load (kW)",
    50,
    8000,
    int(baseline["average_load_kw"])
)

feeder_capacity_kw = st.sidebar.slider(
    "Feeder Capacity (kW)",
    100,
    15000,
    int(baseline["feeder_capacity_kw"])
)

transformer_capacity_kva = st.sidebar.slider(
    "Transformer Capacity (kVA)",
    100,
    15000,
    int(baseline["transformer_capacity_kva"])
)

connected_customers = st.sidebar.slider(
    "Connected Customers",
    100,
    50000,
    int(baseline["connected_customers"])
)

technical_losses_pct = st.sidebar.slider(
    "Technical Losses (%)",
    0.0,
    30.0,
    float(baseline["technical_losses_pct"])
)

voltage_deviation_pct = st.sidebar.slider(
    "Voltage Deviation (%)",
    0.0,
    20.0,
    float(baseline["voltage_deviation_pct"])
)

power_factor = st.sidebar.slider(
    "Power Factor",
    0.50,
    1.00,
    float(baseline["power_factor"])
)

network_type = st.sidebar.selectbox(
    "Network Type",
    ["Radial", "Ring", "Meshed"]
)

# =====================================================
# DERIVED VARIABLES
# =====================================================
transformer_loading_ratio_pct = (
    peak_load_kw / transformer_capacity_kva
) * 100

reverse_power_flow_flag = int(
    proposed_dg > average_load_kw
)

# =====================================================
# BUILD SCENARIO
# =====================================================
scenario = pd.DataFrame({

    "disco_name": [baseline["disco_name"]],
    "feeder_id": [selected_feeder],
    "season": [season],
    "peak_load_kw": [peak_load_kw],
    "average_load_kw": [average_load_kw],
    "load_growth_rate_pct": [baseline["load_growth_rate_pct"]],
    "transformer_capacity_kva": [transformer_capacity_kva],
    "transformer_loading_ratio_pct": [transformer_loading_ratio_pct],
    "feeder_capacity_kw": [feeder_capacity_kw],
    "voltage_deviation_pct": [voltage_deviation_pct],
    "technical_losses_pct": [technical_losses_pct],
    "power_factor": [power_factor],
    "connected_customers": [connected_customers],
    "network_type": [network_type],
    "solar_irradiance_index": [solar_irradiance_index],
    "proposed_dg_kw": [proposed_dg],
    "dg_type": [dg_type],
    "coincidence_factor": [coincidence_factor],
    "reverse_power_flow_flag": [reverse_power_flow_flag]
})

# =====================================================
# ENCODING
# =====================================================
for col, le in encoders.items():
    scenario[col] = le.transform(scenario[col])

X_input = scenario.copy()
X_input = X_input[model.feature_names_in_]

# =====================================================
# PREDICTION
# =====================================================
prediction = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0]
confidence = np.max(proba) * 100

colR1, colR2 = st.columns([1,2])

# =====================================================
# LEFT PANEL
# =====================================================
with colR1:

    #st.subheader("System Hosting Assessment")
    st.markdown(
            "<h1 style='text-align: center; font-size: 20px;color: white ;font-weight: bold;'>"
            "Hosting Assessment</h1>",
            unsafe_allow_html=True
        )
    status_map = {
        0: "SAFE",
        1: "CONSTRAINED",
        2: "CRITICAL"
    }

    status_text = status_map[prediction]

    status_color = {
        0: "#00ff88",
        1: "orange",
        2: "red"
    }[prediction]

    fig_gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=prediction,

        number={
            "font": {
                "size": 16,
                "color": status_color
            },
            "suffix": f"  {status_text}"
        },

        title={
            "text": (
                f"<span style='font-size:18px;'>"
                f"Grid Condition"
                f"</span><br>"
                f"<span style='font-size:18px;'>"
                #f"Confidence: {confidence:.1f}%"
                f"</span>"
            )
        },

        gauge={

            "axis": {
                "range": [0, 2],
                #"tickvals": [0, 1, 2],
                "ticktext": [
                    "SAFE",
                    "CONSTRAINED",
                    "CRITICAL"
                ],
                "tickfont": {
                    "size": 16,
                    "color": "white"
                }
            },

            "bar": {
                "color": "white",
                "thickness": 0.30
            },

            "bgcolor": "#081018",
            "borderwidth": 2,
            "bordercolor": "white",

            "steps": [
                {
                    "range": [0, 0.67],
                    "color": "rgba(0,255,136,0.85)"
                },
                {
                    "range": [0.67, 1.34],
                    "color": "rgba(255,165,0,0.85)"
                },
                {
                    "range": [1.34, 2],
                    "color": "rgba(255,0,0,0.85)"
                }
            ],

            "threshold": {
                "line": {
                    "color": "cyan",
                    "width": 16
                },
                "thickness": 0.9,
                "value": prediction
            }
        }
    ))

    fig_gauge.update_layout(

        paper_bgcolor="#081018",
        plot_bgcolor="#081018",

        font={
            "color": "white",
            "family": "Arial"
        },

        height=400,

        margin=dict(
            l=20,
            r=20,
            t=80,
            b=20
        )
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )


    

# =====================================================
# RIGHT PANEL
# =====================================================
with colR2:

    #st.subheader("AI Explainability")
    st.markdown(
            "<h1 style='text-align: center; font-size: 20px;color: white ;font-weight: bold;'>"
            "AI Explainability</h1>",
            unsafe_allow_html=True
        )
    try:

        X_shap = X_input.copy()
        X_shap_np = X_shap.values

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap_np)

        if isinstance(shap_values, list):
            shap_class_values = shap_values[int(prediction)][0]
            base_values = explainer.expected_value[int(prediction)]
        else:
            shap_array = np.array(shap_values)
            shap_class_values = shap_array[0, :, int(prediction)]

            base_values = (
                explainer.expected_value[int(prediction)]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

        explanation = shap.Explanation(
            values=shap_class_values,
            base_values=base_values,
            data=X_shap_np[0],
            feature_names=X_input.columns
        )

        fig, ax = plt.subplots(
            figsize=(10, 18),
            facecolor='#081018'
        )

        ax.set_facecolor('#081018')

        shap.plots.waterfall(explanation, show=False)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error("SHAP explanation could not be generated.")
        st.exception(e)


#st.markdown("### AI Interpretation")

if prediction == 0:
    st.success(
        "The network can SAFELY host the proposed DG under current operating conditions."
    )

elif prediction == 1:
    st.warning(
        "The network is approaching OPERATIONAL CONSTRAINTS and may require mitigation measures."
    )

else:
    st.error(
        "The proposed DG level may introduce SIGNIFICANT operational RISKS including overloads and reverse power flow."
    )

st.divider()
# =====================================================
# EXECUTIVE KPI CARDS
# =====================================================
#st.subheader("Operational Intelligence")
st.markdown(
        "<h1 style='text-align: left; font-size: 20px;color: white ;font-weight: bold;'>"
        "Operational Intelligence</h1>",
        unsafe_allow_html=True
    )
available_capacity = feeder_capacity_kw - peak_load_kw
dg_penetration_pct = (proposed_dg / peak_load_kw) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Available Capacity
            </div>
            <div class="metric-value">
                {available_capacity:.1f} kW
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                DG Penetration
            </div>
            <div class="metric-value">
                {dg_penetration_pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Transformer Loading
            </div>
            <div class="metric-value">
                {transformer_loading_ratio_pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.divider()
# =====================================================
# HOSTING CURVE
# =====================================================
#st.subheader("Hosting Capacity Sensitivity Curve")
st.markdown(
        "<h1 style='text-align: left; font-size: 20px;color: white ;font-weight: bold;'>"
        "Hosting Sensitivity Curve</h1>",
        unsafe_allow_html=True
    )
dg_range = np.arange(0, 5000, 250)

results = []

for dg in dg_range:

    temp = scenario.copy()

    temp["proposed_dg_kw"] = dg

    temp["reverse_power_flow_flag"] = int(
        dg > average_load_kw
    )

    temp = temp[model.feature_names_in_]

    pred = model.predict(temp)[0]

    results.append({
        "DG_kW": dg,
        "Prediction": pred
    })

curve_df = pd.DataFrame(results)

fig_curve = px.area(
    curve_df,
    x="DG_kW",
    y="Prediction",
    markers=True
)

fig_curve.update_traces(
    line=dict(width=4),
    opacity=0.85
)

fig_curve.update_layout(

    template="plotly_dark",

    paper_bgcolor="#081018",
    plot_bgcolor="#081018",

    font=dict(
        color="white",
        size=14
    ),

    hovermode="x unified",

    height=500,

    title={
        "text": "Grid Capacity Response",
        "x": 0.5,
        "font": {
            "size": 24
        }
    }
)

fig_curve.update_yaxes(
    tickvals=[0, 1, 2],
    ticktext=[
        "SAFE",
        "CONSTRAINED",
        "CRITICAL"
    ],
    title="Hosting Status"
)

fig_curve.update_xaxes(
    title="Distributed Generation Capacity (kW)"
)

fig_curve.add_vline(
    x=proposed_dg,
    line_dash="dash",
    line_color="cyan",
    line_width=3,
    annotation_text=f"Selected DG = {proposed_dg} kW"
)

st.plotly_chart(
    fig_curve,
    use_container_width=True
)

st.divider()

# =====================================================
# BASELINE FEEDER SNAPSHOT
# =====================================================
st.subheader("Baseline Feeder Snapshot")

st.dataframe(
    feeder_df.tail(5)
)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.caption(
    "AI-Powered Grid Hosting Capacity Assessment Platform"
)
