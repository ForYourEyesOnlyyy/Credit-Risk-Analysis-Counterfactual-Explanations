import streamlit as st
import yaml
from src.loan_grader import predict_loan_grade
from src.model import get_model, predict
from src.counterfactual_explanations import counterfactual_explanation

st.set_page_config(page_title="Loan Risk Assessment", layout="wide")
st.title("🏦 Loan Risk Assessment")

# Load config
with open("configs/features.yaml", "r") as f:
    config = yaml.safe_load(f)

numerical_config = config["numerical_features"]
categorical_config = config["categorical_features"]


@st.cache_resource
def load_default_model():
    return get_model()  # returns (model, device)

with st.spinner("Loading model..."):
    default_model, device = load_default_model()
st.success("Model loaded successfully!")

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

if "default_result" not in st.session_state:
    st.session_state.default_result = None
if "cf_input" not in st.session_state:
    st.session_state.cf_input = None

# 🔐 Predict Default on Button Click
if st.button("🚨 Predict Loan Default"):
    try:
        default_result = predict(default_model, inputs)
        st.session_state.default_result = default_result

        if default_result == 1:
            status = "❌ Likely to Default"
            color = "red"
        else:
            status = "✅ Likely to be Repaid"
            color = "green"

        st.markdown(f"<div style='text-align: center; font-size: 24px; color: {color};'>{status}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Default prediction failed: {e}")
        st.session_state.default_result = None

with open("configs/data.yaml", "r") as f:
    data_config = yaml.safe_load(f)

optimizable_keys = (
    data_config["optimizable_numerical_columns"]
    + data_config["optimizable_ordinal_columns"]
    + data_config["optimizable_categories"]
)

def print_cf_changes(original: dict, counterfactual: dict, fields: list, tol=1e-6):
    st.markdown("### 🔄 Counterfactual Suggestion")
    st.markdown("Below are the fields the model suggests changing to avoid default:")

    changes_found = False
    for key in fields:
        orig_val = original.get(key)
        cf_val = counterfactual.get(key)

        if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
            changed = abs(float(orig_val) - float(cf_val)) > tol
        else:
            changed = str(orig_val) != str(cf_val)

        if changed:
            changes_found = True
            st.write(f"**{key.replace('_', ' ').capitalize()}**: `{orig_val}` → `{cf_val}`")

    if not changes_found:
        st.success("✅ No changes needed — this input is already optimized.")


# 🤖 Counterfactual Explanation
if st.session_state.default_result == 1:
    st.markdown("### 🧠 This loan is risky — want to explore how to make it safer?")

    if st.button("🔍 Explain"):
        with st.spinner("Generating counterfactual..."):
            try:
                cf_input = counterfactual_explanation(inputs, 1)
                st.session_state.cf_input = cf_input
                st.success("Counterfactual generated!")
            except Exception as e:
                st.error(f"Explanation failed: {e}")

# 📊 Comparison Table
if st.session_state.cf_input:
    cf = st.session_state.cf_input
    orig = inputs

    st.markdown("### 🔄 Counterfactual Suggestion")
    st.markdown("This is an alternative version of the loan application that would be predicted as **not defaulting**.")

    st.write("**Changes suggested:**")
    print_cf_changes(orig, cf, optimizable_keys)