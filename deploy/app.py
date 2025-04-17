import streamlit as st
import yaml
from src.loan_grader import predict_loan_grade
from src.model import get_model, predict

st.set_page_config(page_title="Loan Risk Assessment", layout="wide")
st.title("🏦 Loan Risk Assessment")

# Load config
with open("configs/features.yaml", "r") as f:
    config = yaml.safe_load(f)

numerical_config = config["numerical_features"]
categorical_config = config["categorical_features"]

try:
    default_model, device = get_model()
except RuntimeError as e:
    st.warning("⚠️ Known PyTorch issue during reload — safe to ignore.")
    default_model, device = get_model()

# Split layout
col1, col2 = st.columns(2)
inputs = {}

num_keys = list(numerical_config.keys())
half = len(num_keys) // 2

with col1:
    for key in num_keys[:half]:
        feature = numerical_config[key]
        dtype = feature["type"]
        step = 1 if dtype == "int" else 0.01
        inputs[key] = st.number_input(
            label=feature["description"],
            min_value=feature["min"],
            max_value=feature["max"],
            value=feature["min"],
            step=step,
            help=key
        )

with col2:
    for key in num_keys[half:]:
        feature = numerical_config[key]
        dtype = feature["type"]
        step = 1 if dtype == "int" else 0.01
        inputs[key] = st.number_input(
            label=feature["description"],
            min_value=feature["min"],
            max_value=feature["max"],
            value=feature["min"],
            step=step,
            help=key
        )

# Collect categorical inputs
cat_keys = list(categorical_config.keys())
half_cat = len(cat_keys) // 2

with col1:
    for key in cat_keys[:half_cat]:
        feature = categorical_config[key]
        inputs[key] = st.selectbox(
            label=feature["description"],
            options=feature["allowed_values"],
            help=key
        )

with col2:
    for key in cat_keys[half_cat:]:
        feature = categorical_config[key]
        inputs[key] = st.selectbox(
            label=feature["description"],
            options=feature["allowed_values"],
            help=key
        )

# 🔢 Calculate derived feature: loan_percent_income
try:
    if inputs["person_income"] > 0:
        inputs["loan_percent_income"] = inputs["loan_amnt"] / inputs["person_income"]
    else:
        inputs["loan_percent_income"] = 0.0
except Exception as e:
    st.error(f"Failed to calculate loan_percent_income: {e}")
else:
    st.markdown(f"**📊 Calculated Loan-to-Income Ratio:** {round(inputs['loan_percent_income']*100, 2)}")

# 🎯 Predict Loan Grade (live)
try:
    grade = predict_loan_grade(inputs)
    inputs["loan_grade"] = grade
except Exception as e:
    inputs["loan_grade"] = None
    st.error(f"Failed to calculate loan grade: {e}")
else:
    st.markdown(f"**🏷️ Predicted Loan Grade:** `{grade}`")

# 🔐 Predict Default on Button Click
if st.button("🚨 Predict Loan Default"):
    try:
        default_result = predict(default_model, inputs)
        status = "❌ Likely to Default" if default_result == 1 else "✅ Likely to be Repaid"
        color = "red" if default_result == 1 else "green"
        st.markdown(f"<div style='text-align: center; font-size: 24px; color: {color};'>{status}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Default prediction failed: {e}")
