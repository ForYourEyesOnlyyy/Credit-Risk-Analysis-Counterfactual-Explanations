import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import yaml
import os 
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv('PYTHONPATH')

loan_grader_config_path = f'{project_root}/configs/loan_grader.yaml'

def load_loan_grader_config():
    with open(loan_grader_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_loan_grade_model(save_model=True):
    """
    Train a Random Forest classifier to predict loan grades using optimal parameters.
    
    Args:
        save_model (bool): Whether to save the trained model to disk
        
    Returns:
        tuple: (model, X_test, y_test) - model and test data for evaluation
    """
    # Load the processed data
    df = pd.read_csv('data/processed/train.csv')
    
    # Prepare features - adding more relevant features
    base_features = [
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'cb_person_default_on_file_Y',
        'person_emp_length',
        'person_age',
        'person_income',
        'loan_amnt'
    ]
    
    # Create interaction features
    df['income_to_credit_length'] = df['person_income'] * df['cb_person_cred_hist_length']
    df['loan_to_credit_length'] = df['loan_amnt'] * df['cb_person_cred_hist_length']
    df['income_to_age'] = df['person_income'] / (df['person_age'] + 1)  # +1 to avoid division by zero
    
    # Add interaction features
    interaction_features = [
        'income_to_credit_length',
        'loan_to_credit_length',
        'income_to_age'
    ]
    
    # Combine all features
    features = base_features + interaction_features
    
    X = df[features].copy()
    
    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Get actual loan grades from one-hot encoded columns
    grade_columns = [col for col in df.columns if col.startswith('loan_grade_')]
    y = []
    for _, row in df[grade_columns].iterrows():
        grade = row[row == 1.0].index[0].split('_')[-1]
        y.append(grade)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train Random Forest with optimal parameters
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    rf.fit(X_train, y_train)
    
    if save_model:
        # Create models directory if it doesn't exist
        model_dir = Path('models/loan_grade_rf')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump(rf, model_dir / 'model.joblib')
        
        # Save feature names and scaler
        joblib.dump({'features': features, 'scaler': scaler}, model_dir / 'preprocessing.joblib')
    
    return rf, X_test, y_test

def predict_loan_grade(
    loan_int_rate: float,
    loan_percent_income: float,
    cb_person_cred_hist_length: int,
    cb_person_default_on_file: str,
    person_emp_length: int,
    person_age: int = 30,  # Default values for new parameters
    person_income: float = 50000,
    loan_amnt: float = 10000
) -> str:
    """
    Predict loan grade using the trained Random Forest model.
    
    Args:
        loan_int_rate: Interest rate (%)
        loan_percent_income: Loan amount to income ratio (%)
        cb_person_cred_hist_length: Credit history length in years
        cb_person_default_on_file: Previous defaults ('Y' or 'N')
        person_emp_length: Employment length in years
        person_age: Age of the person
        person_income: Annual income
        loan_amnt: Loan amount
        
    Returns:
        str: Predicted loan grade ('A' through 'G')
    """
    config = load_loan_grader_config()
    # Load the model and preprocessing info
    model = joblib.load(config['model'])
    preprocessing = joblib.load(config['preprocessing'])
    features = preprocessing['features']
    scaler = preprocessing['scaler']
    
    # Scale inputs
    person_income_scaled = person_income / 6000000  # Max income in dataset
    loan_amnt_scaled = loan_amnt / 35000  # Max loan amount in dataset
    
    # Create interaction features
    income_to_credit_length = person_income_scaled * (cb_person_cred_hist_length / 30)
    loan_to_credit_length = loan_amnt_scaled * (cb_person_cred_hist_length / 30)
    income_to_age = person_income_scaled / ((person_age / 85) + 1)  # Max age was 85
    
    # Prepare features
    features_dict = {
        'loan_int_rate': loan_int_rate / 23.22,
        'loan_percent_income': loan_percent_income / 100,
        'cb_person_cred_hist_length': cb_person_cred_hist_length / 30,
        'cb_person_default_on_file_Y': 1 if cb_person_default_on_file == 'Y' else 0,
        'person_emp_length': person_emp_length / 50,
        'person_age': person_age / 85,
        'person_income': person_income_scaled,
        'loan_amnt': loan_amnt_scaled,
        'income_to_credit_length': income_to_credit_length,
        'loan_to_credit_length': loan_to_credit_length,
        'income_to_age': income_to_age
    }
    
    # Create DataFrame with features in correct order
    X = pd.DataFrame([features_dict])[features]
    
    # Scale features
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    
    # Make prediction
    return model.predict(X_scaled)[0]

if __name__ == "__main__":
    # Train the model and print results
    rf_model, X_test, y_test = train_loan_grade_model() 