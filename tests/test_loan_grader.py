import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.loan_grader import train_loan_grade_model, predict_loan_grade

def evaluate_model():
    """
    Train and evaluate the Random Forest model for loan grade prediction.
    Prints detailed performance metrics and saves the model.
    
    Returns:
        float: Model accuracy on test set
    """
    # Train model and get test data
    rf_model, X_test, y_test = train_loan_grade_model()
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importances
    features = [
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'cb_person_default_on_file_Y',
        'person_emp_length'
    ]
    feature_importances = dict(zip(features, rf_model.feature_importances_))
    
    # Print detailed results
    print("\nLoan Grade Prediction Results (Random Forest):")
    print(f"Test set accuracy: {accuracy:.2%}")
    
    print("\nFeature Importances:")
    for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    grade_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    conf_df = pd.DataFrame(
        conf_matrix, 
        index=[f'True_{g}' for g in grade_options],
        columns=[f'Pred_{g}' for g in grade_options]
    )
    print(conf_df)
    
    return accuracy

def test_prediction_interface():
    """
    Test the prediction interface with a sample loan application.
    """
    # Test case
    test_input = {
        'loan_int_rate': 12.5,
        'loan_percent_income': 25.0,
        'cb_person_cred_hist_length': 5,
        'cb_person_default_on_file': 'N',
        'person_emp_length': 4
    }
    
    # Make prediction
    grade = predict_loan_grade(**test_input)
    
    print("\nTest Prediction:")
    print(f"Input: {test_input}")
    print(f"Predicted Grade: {grade}")
    
    # Basic validation
    assert grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G'], "Invalid grade prediction"

if __name__ == "__main__":
    print("Evaluating Random Forest model performance...")
    accuracy = evaluate_model()
    
    print("\nTesting prediction interface...")
    test_prediction_interface() 