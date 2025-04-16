import sys
import os

from src.model import get_model, predict
import torch

def test_model():
    print("\n=== Testing Model Loading ===")
    # Get the model and its device
    model, device = get_model()
    print(f"Model loaded successfully on device: {device}")
    print(f"Model architecture: {model.model}")

    # Test input data
    test_input = [
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

    print("\n=== Testing Model Prediction ===")
    print("Test input:", test_input)
    
    # Make prediction
    prediction = predict(model, test_input)
    print(f"Prediction: {prediction} (1 = default, 0 = non-default)")

if __name__ == "__main__":
    test_model() 