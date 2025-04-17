import yaml
import joblib

config_path = 'configs/models.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def get_scaler():
    scaler_path = config['scaler_path']
    scaler = joblib.load(scaler_path)
    return scaler