from src.model import get_model, predict
import torch

def test_model():
    print("\n=== Testing Model Loading ===")
    model, device = get_model()
    print(f"Model loaded successfully on device: {device}")
    print(f"Model architecture:\n{model.model}")

    # Test input data
    test_input = {
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

    print("\n=== Testing Model Prediction ===")
    print("Test input:", test_input)
    
    # Make prediction
    prediction = predict(model, test_input)
    print(f"Prediction: {prediction} (1 = default, 0 = non-default)")

if __name__ == "__main__":
    test_model() 