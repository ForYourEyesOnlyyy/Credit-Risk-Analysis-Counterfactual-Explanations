import joblib
import yaml

config_path = 'configs/models.yaml'
data_config = 'configs/data.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

with open(data_config, 'r') as f:
    data_config = yaml.safe_load(f)

def predict_loan_grade(
    data: dict
) -> str:
    """
    Predict loan grade using the trained Random Forest model.
    
    Args:
        data: vocab of data
        
    Returns:
        str: Predicted loan grade ('A' through 'G')
    """
    input_list = [data[feature] for feature in data_config['loan_grading']['features']]

    lg = joblib.load(config['loan_grader_model_path'])
    prediction = lg.predict([input_list])[0]
    return data_config['loan_grading']['grades'][prediction]

if __name__ == "__main__":
    test_input = {
        'person_income': 59000,
        'loan_amnt': 35000, 
        'loan_int_rate': 16.02, 
        'loan_percent_income': 0.59
    }
    result = predict_loan_grade(test_input) 