import yaml
import joblib

config_path = 'configs/models.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


def get_encoder():
    encoder_path = config['one_hot_encoder_path']
    with open(encoder_path, 'rb') as file:
        return joblib.load(file)

