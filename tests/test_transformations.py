from src.data import transform, inverse_transform

# First line from the raw data (without target column):
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

def compare_dicts(original: dict, recovered: dict):
    print("\n🔁 Comparison of Original vs Inverse Transformed:")
    for key in original:
        orig_val = original[key]
        inv_val = recovered.get(key, None)
        match = "✓" if str(orig_val) == str(inv_val) else "✗"
        print(f"{match} {key}: {orig_val} -> {inv_val}")

def test_transform_inverse_cycle():
    print("Original raw input:\n", raw_input)
    
    # Transform raw input to model format
    transformed = transform(raw_input)
    print("\nTransformed shape:", transformed.shape)
    print("Transformed values:", transformed)
    
    # Transform back to raw format
    raw_output = inverse_transform(transformed)
    print("\nInverse transformed output:", raw_output)
    
    # Compare input and output
    compare_dicts(raw_input, raw_output)

if __name__ == "__main__":
    test_transform_inverse_cycle()


