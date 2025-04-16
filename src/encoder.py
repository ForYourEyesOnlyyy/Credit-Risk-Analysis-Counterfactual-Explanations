import yaml
import os
from dotenv import load_dotenv
import joblib

load_dotenv()
project_root = os.getenv('PYTHONPATH')
encoder_path = f'{project_root}/models/encoder/encoder.pkl'


def get_encoder_config():
    with open(encoder_path, 'r') as file:
        return yaml.safe_load(file)


def get_encoder():
    with open(encoder_path, 'rb') as file:
        return joblib.load(file)

