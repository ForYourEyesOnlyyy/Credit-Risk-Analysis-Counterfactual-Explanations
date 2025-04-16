import sys
import os

from src.counterfactual_explanations import counterfactual_explanation

def test_counterfactual_explanation():
    # Raw input data
    raw_input = [
        22,                     # person_age
        59000,                  # person_income
        'RENT',                 # person_home_ownership
        123.0,                  # person_emp_length
        'PERSONAL',             # loan_intent
        'D',                    # loan_grade
        35000,                  # loan_amnt
        16.02,                  # loan_int_rate
        0.59,                   # loan_percent_income
        'Y',                    # cb_person_default_on_file
        3                       # cb_person_cred_hist_length
    ]

    print("\n=== Original Input ===")
    print("Raw input:", raw_input)

    # Generate counterfactual explanation
    print("\n=== Counterfactual Generation ===")
    # Assuming original prediction was 1 (default), so we want to find a counterfactual for 0 (non-default)
    raw_counterfactual = counterfactual_explanation(raw_input, y_original=1)
    print("Raw counterfactual:", raw_counterfactual)

    # Print detailed comparison
    print("\n=== Detailed Comparison ===")
    print("Feature\t\tOriginal\tCounterfactual\tChange")
    print("-" * 60)
    for i, (orig, cf) in enumerate(zip(raw_input, raw_counterfactual)):
        change = "✓" if str(orig) == str(cf) else "✗"
        feature_name = [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length"
        ][i]
        print(f"{feature_name:<20} {str(orig):<15} {str(cf):<15} {change}")

if __name__ == "__main__":
    test_counterfactual_explanation() 