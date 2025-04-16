from sklearn.preprocessing import MinMaxScaler
import yaml
import joblib
from dotenv import load_dotenv
import os

load_dotenv()
project_root = os.getenv('PYTHONPATH')

scaler_config_path = f'{project_root}/configs/scaler.yaml'

def get_scaler_config():
    with open(scaler_config_path, 'r') as file:
        return yaml.safe_load(file)

def get_scaler():
    scaler_config = get_scaler_config()
    scaler = joblib.load(scaler_config['path']) 
    return scaler