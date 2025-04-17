from src.loan_grader import predict_loan_grade


def test_prediction_interface():
    """
    Test the prediction interface with a sample loan application.
    """
    # Test case
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
    
    # Make prediction
    grade = predict_loan_grade(test_input)
    
    print("\nTest Prediction:")
    print(f"Input: {test_input}")
    print(f"Predicted Grade: {grade}")
    
    # Basic validation
    assert test_input['loan_grade'] == grade, "Invalid grade prediction"

if __name__ == "__main__":
    
    print("\nTesting prediction interface...")
    test_prediction_interface() 