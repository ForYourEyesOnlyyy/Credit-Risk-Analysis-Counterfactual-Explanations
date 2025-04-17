import sys
import os

from src.counterfactual_explanations import counterfactual_explanation

# Raw input data
raw_input = {
        'person_age': 22,
        'person_income': 59000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 123.0,
        'loan_intent': 'PERSONAL',
        'loan_grade': 'D',
        'loan_amnt': 35000,
        'loan_int_rate': 16.02,
        'loan_percent_income': 0.59,
        'cb_person_default_on_file': 'Y',
        'cb_person_cred_hist_length': 3
    }

def compare_counterfactuals(raw_input: dict, raw_counterfactual: dict, tol=1e-6):
    print("\n=== Detailed Comparison ===")
    print(f"{'Feature':<25} {'Original':<15} {'Counterfactual':<15} {'Change'}")
    print("-" * 70)

    for key in raw_input.keys():
        orig = raw_input.get(key, None)
        cf = raw_counterfactual.get(key, None)

        if isinstance(orig, (int, float)) and isinstance(cf, (int, float)):
            changed = "✓" if abs(float(orig) - float(cf)) > tol else "✗"
        else:
            changed = "✓" if str(orig) != str(cf) else "✗"

        print(f"{key:<25} {str(orig):<15} {str(cf):<15} {changed}")



def test_counterfactual_explanation():
    print("\n=== Original Input ===")
    print("Raw input:", raw_input)

    # Generate counterfactual explanation
    print("\n=== Counterfactual Generation ===")
    # Assuming original prediction was 1 (default), so we want to find a counterfactual for 0 (non-default)
    raw_counterfactual = counterfactual_explanation(raw_input, y_original=1)
    print("Raw counterfactual:", raw_counterfactual)

    # Print detailed comparison
    compare_counterfactuals(raw_input, raw_counterfactual)
  

if __name__ == "__main__":
    test_counterfactual_explanation() 