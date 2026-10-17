"""Run with: python -m streamlit run app_rf.py"""
from __future__ import annotations

import csv
import io
import json
from html import escape
import streamlit as st

from rf_inference import FEATURE_LABELS, evaluate_patient, load_model
from rf_visuals import force_plot, figure_bytes

st.set_page_config(page_title="RF-based dNCR Prediction", page_icon="🧠", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background: #FFFFFF; color: #262730; }
.block-container { max-width: 1040px; padding-top: 2.5rem; padding-bottom: 3rem; }
h1 { font-size: clamp(2rem, 4.4vw, 3.05rem) !important; line-height: 1.14 !important;
     font-weight: 750 !important; letter-spacing: -.035em; margin-bottom: 1.25rem !important; }
h2 { letter-spacing: -.02em; }
[data-testid="stWidgetLabel"] p { font-size: 1rem; font-weight: 500; }
[data-testid="stNumberInputContainer"], [data-baseweb="select"] > div {
    background: #F0F2F6 !important; border: 1px solid transparent !important;
    border-radius: 10px !important; min-height: 50px; }
[data-testid="stNumberInputContainer"]:focus-within { border-color: #AAB4C4 !important; }
[data-testid="stNumberInput"] input { font-size: 1.06rem; }
[data-testid="stNumberInput"] button { background: transparent !important; }
.stButton > button { min-height: 49px; border-radius: 10px; font-size: 1.05rem; }
.stButton > button:hover { border-color: #364B6B; color: #253D5E; background: #F7F9FC; }
.st-key-predict { padding-top: .45rem; padding-bottom: 1.15rem; }
.result-row { margin: 1.05rem 0; font-size: 1.08rem; color: #414751; }
.result-row strong { font-weight: 650; }
.risk-box { padding: 1rem 1.2rem; border-radius: 9px; margin: 1.2rem 0 1.6rem;
            font-size: 1.1rem; font-weight: 600; }
.risk-low { background: #E8F8EE; color: #29884F; }
.risk-high { background: #FFF0F1; color: #BD3150; }
.subtle { color: #727987; font-size: .92rem; }
[data-testid="stCaptionContainer"] p { color: #6B7280; }
@media (max-width: 640px) { .block-container { padding-left: 1.1rem; padding-right: 1.1rem; }
    h1 { font-size: 2rem !important; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cached_model():
    return load_model("RF")


def clear_result():
    st.session_state.pop("prediction_result", None)


with st.sidebar:
    st.header("Model settings")
    st.write("Random Forest (RF)")
    operating_point = st.selectbox("Operating point", ["primary", "exploratory_90"], format_func=lambda x: "Primary analysis" if x == "primary" else "Exploratory high sensitivity", key="operating_point", on_change=clear_result)
    st.caption("Changing the operating point changes the risk classification, not the predicted probability.")
    if operating_point == "exploratory_90":
        st.warning("Post-hoc exploration. This threshold requires validation in a new cohort.")
    st.divider()
    st.caption("Five-variable RF model\n\ndNCR-RF-20261017-v2")

model = cached_model()
previous = st.session_state.get("prediction_result")
if previous and (previous.get("model") != "RF" or previous.get("version") != model["version"]):
    clear_result()
st.title("RF-based dNCR Prediction")
st.caption("Random Forest (RF) · RF-only v3")
st.write("Please enter patient characteristics:")

for field, default in {"age": 70.0, "moca": 25.0, "duration": 120.0, "gfr": 80.0}.items():
    if field not in st.session_state:
        st.session_state[field] = default
age = st.number_input("Age:", min_value=0.0, max_value=120.0, value=None, step=1.0, format="%.2f", help="Age in years. The development cohort included patients aged 60–90 years.", key="age", on_change=clear_result)
moca = st.number_input("MOCA Score:", min_value=0.0, max_value=30.0, value=None, step=1.0, format="%.2f", key="moca", on_change=clear_result)
duration = st.number_input("Operation Time (minutes):", min_value=1.0, max_value=1440.0, value=None, step=1.0, format="%.2f", key="duration", on_change=clear_result)
gfr = st.number_input("GFR:", min_value=0.0, max_value=300.0, value=None, step=1.0, format="%.2f", help="Use the same GFR definition and unit as the original training data.", key="gfr", on_change=clear_result)
depression_label = st.selectbox("Depression:", ["No", "Yes"], index=0, key="depression", on_change=clear_result)

patient = {"Age": age, "MOCA_Score": moca, "Operation_Time": duration, "GFR": gfr, "Depression": int(depression_label == "Yes")}
if st.button("Predict", width="stretch", key="predict"):
    try:
        st.session_state["prediction_result"] = evaluate_patient(model, patient, operating_point)
    except (ValueError, ArithmeticError) as exc:
        clear_result()
        st.error(str(exc))

result = st.session_state.get("prediction_result")
if result is not None:
    st.divider()
    if result["outside_training_range"]:
        st.warning("Outside the observed development-data range: " + ", ".join(result["outside_training_range"]) + ". This prediction may not generalize to this input profile.")
    if result["clipped_features"]:
        st.caption("Training-derived preprocessing bounds were applied to: " + ", ".join(result["clipped_features"]) + ". See the input details below.")
    if result["operating_point"] != "primary":
        st.warning("Exploratory operating point — post-hoc threshold; prospective validation is required.")
    probability = result["probability"]
    st.markdown(f'<div class="result-row">Predicted probability: <strong>{probability:.4f}</strong> ({probability:.2%})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-row">Decision threshold: <strong>{result["threshold"]:.6f}</strong></div>', unsafe_allow_html=True)
    predicted = "Yes" if result["predicted_class"] else "No"
    st.markdown(f'<div class="result-row">Predicted result: <strong>{predicted}</strong></div>', unsafe_allow_html=True)
    risk_class = "risk-high" if result["predicted_class"] else "risk-low"
    risk_text = "Higher predicted risk of dNCR" if result["predicted_class"] else "Lower predicted risk of dNCR"
    st.markdown(f'<div class="risk-box {risk_class}">{escape(risk_text)}</div>', unsafe_allow_html=True)
    st.subheader("SHAP Force Plot")
    figure = force_plot(result)
    png_bytes = figure_bytes(figure, "png")
    st.image(png_bytes, width="stretch")
    st.caption("Red pushes the prediction higher; blue pushes it lower, relative to the fixed reference profile. Numbers link the arrows to the five inputs.")
    with st.expander("Input values and SHAP contributions"):
        rows = []
        for i, feature in enumerate(result["features"]):
            rows.append({"Feature": FEATURE_LABELS[feature], "Entered value": result["inputs"][i], "Model input": result["model_inputs"][i], "Reference value": result["reference_profile"][i], "SHAP value": result["shap_values"][i]})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(f'Base value {result["base_value"]:.6f} + sum of SHAP values {sum(result["shap_values"]):+.6f} = predicted probability {probability:.6f}.')
        st.caption("The reference profile is the average of the five imputed development-cohort mean profiles. It is an aggregate comparator, not an individual patient or an estimate of population prevalence. SHAP contributions explain the model, not causal effects.")
    with st.expander("Download result and plot"):
        row = {"Model": result["model"], "Version": result["version"], "Operating_point": result["operating_point"], "Probability_dNCR": probability, "Threshold": result["threshold"], "Predicted_class": result["predicted_class"], "SHAP_base_value": result["base_value"]}
        for i, feature in enumerate(result["features"]):
            row[feature] = result["inputs"][i]
            row[feature+"_model_input"] = result["model_inputs"][i]
            row[feature+"_SHAP"] = result["shap_values"][i]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        st.download_button("Download prediction (CSV)", stream.getvalue().encode("utf-8-sig"), file_name="dNCR_RF_prediction.csv", mime="text/csv", key="download_csv", on_click="ignore")
        columns = st.columns(3)
        for column, kind, mime in zip(columns, ("png", "pdf", "tiff"), ("image/png", "application/pdf", "image/tiff")):
            column.download_button(kind.upper(), png_bytes if kind == "png" else figure_bytes(figure, kind), file_name=f"dNCR_RF_SHAP_force_plot.{kind}", mime=mime, key="download_"+kind, width="stretch", on_click="ignore")
        st.download_button("Download full result (JSON)", json.dumps(result, indent=2), file_name="dNCR_RF_prediction.json", mime="application/json", key="download_json", on_click="ignore")

with st.expander("About this RF model"):
    metrics = model["external_metrics"]
    st.write("Predictors: Age, MOCA Score, Operation Time, GFR and Depression. Depression is coded No = 0 and Yes = 1; dNCR = 1 is the modeled event.")
    st.write(f'Primary-analysis temporal external validation: AUC {metrics["AUC"]:.3f} (95% CI {metrics["AUC_low"]:.3f}–{metrics["AUC_high"]:.3f}), sensitivity {metrics["Sensitivity"]:.3f}, specificity {metrics["Specificity"]:.3f}. These metrics refer to the primary threshold.')
    st.write("The primary threshold was determined from development-data cross-validation. The optional high-sensitivity threshold was explored after reviewing the main external results.")
    st.write("The inputs must be complete. Operation Time means this model is used at or after the end of surgery. Confirm variable definitions and units before using the calculator.")
    st.caption("Exact SHAP values are computed over all 32 subsets of the five predictors using one aggregate reference profile. This reference differs from the background sample used for the global SHAP plots in the analysis report.")
st.caption("For research use only. Model predictions are not a clinical diagnosis. Input values are processed on the hosting server and are not written to an application database.")
