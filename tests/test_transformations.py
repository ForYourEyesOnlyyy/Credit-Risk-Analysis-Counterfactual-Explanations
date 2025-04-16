from src.data import transform, inverse_transform

# First line from the raw data (without target column):
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

def test_transform_inverse_cycle():
    print("Original raw input:", raw_input)
    
    # Transform raw input to model format
    transformed = transform(raw_input)
    print("\nTransformed shape:", transformed.shape)
    print("Transformed values:", transformed)
    
    # Transform back to raw format
    raw_output = inverse_transform(transformed)
    print("\nInverse transformed output:", raw_output)
    
    # Compare input and output
    print("\nComparison:")
    for i, (orig, inv) in enumerate(zip(raw_input, raw_output)):
        match = "✓" if str(orig) == str(inv) else "✗"
        print(f"{match} {orig} -> {inv}")

if __name__ == "__main__":
    test_transform_inverse_cycle()


