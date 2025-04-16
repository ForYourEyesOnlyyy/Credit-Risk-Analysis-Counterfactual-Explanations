'''
This file contains functions for processing the data.

- scale_transform_X
- inverse_scale_transform_X

- generate_mask
- get_ordinal_feature_names
- get_categorical_to_optimize
- get_recalc_params_matrix
- build_one_hot_groups
- filter_one_hot_groups
- form_recalc_dict
'''


import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

load_dotenv()

from src.scaler import get_scaler
from src.encoder import get_encoder

project_root = os.getenv('PYTHONPATH')

data_config_path = f'{project_root}/configs/data.yaml'

def get_data_config():
    with open(data_config_path, 'r') as file:
        return yaml.safe_load(file)

def transform(raw_input: list) -> torch.Tensor:
    '''
    Transform raw input values into model-ready tensor.
    Handles numerical scaling and categorical encoding.
    
    Args:
        raw_input: List of raw values in the order defined by config['columns']
    
    Returns:
        torch.Tensor: Transformed values ready for model input
    '''
    config = get_data_config()
    scaler = get_scaler()
    encoder = get_encoder()

    df = pd.DataFrame([raw_input], columns=config['raw_columns'])

    numerical_cols = config['all_numerical_columns']
    X_numerical = df[numerical_cols]

    X_numerical_scaled = scaler.transform(X_numerical)

    ordinal_cols = config['all_ordinal_columns']
    X_ordinal = df[ordinal_cols]

    X_ordinal = X_ordinal['person_home_ownership'].replace(
        {'OTHER': 0, 'RENT': 1, 'MORTGAGE': 2, 'OWN': 3}
    ).astype(float).values.reshape(-1, 1)

    categorical_cols = config['base_categories']
    X_categorical = df[categorical_cols]

    X_categorical_encoded = encoder.transform(X_categorical).toarray()
    
    X_tensor = torch.tensor(np.concatenate([X_numerical_scaled, X_ordinal, X_categorical_encoded], axis=1), dtype=torch.float)
    return X_tensor
   


def inverse_transform(X_tensor):
    '''
    Inverse transform the model output tensor back to raw input format.
    Takes a tensor in the format [numerical_scaled + ordinal + one_hot] and
    converts it back to raw values in the order defined by config['raw_columns'].
    '''
    config = get_data_config()
    scaler = get_scaler()
    encoder = get_encoder()

    # Convert tensor to numpy
    X = X_tensor.detach().cpu().numpy()
    all_cols = config['columns']
    
    # Get numerical values using column indices
    numerical_indices = [all_cols.index(col) for col in config['all_numerical_columns']]
    X_numerical = X[:, numerical_indices]
    X_numerical_raw = scaler.inverse_transform(X_numerical)
    
    # Get ordinal value using column index
    ordinal_idx = all_cols.index('person_home_ownership')
    ordinal_mapping = {0: 'OTHER', 1: 'RENT', 2: 'MORTGAGE', 3: 'OWN'}
    X_ordinal_raw = ordinal_mapping[int(round(float(X[:, ordinal_idx])))]
    
    # Get all categorical one-hot features at once
    categorical_indices = []
    for cat in config['base_categories']:
        cat_indices = [i for i, col in enumerate(all_cols) if col.startswith(f'{cat}_')]
        categorical_indices.extend(cat_indices)
    X_categorical = X[:, categorical_indices]
    
    # Convert one-hot back to categories all at once
    X_categorical_raw = encoder.inverse_transform(X_categorical)
    
    # Combine all back together in the order of raw_columns
    result = []
    categorical_idx = 0
    for col in config['raw_columns']:
        if col in config['all_numerical_columns']:
            idx = config['all_numerical_columns'].index(col)
            val = X_numerical_raw[0, idx]
            # Round to appropriate precision based on the column
            if col in ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']:
                result.append(int(round(val)))
            else:  # For float columns like loan_int_rate and loan_percent_income
                result.append(float(f"{val:.2f}"))  # Format to exactly 2 decimal places
        elif col in config['all_ordinal_columns']:
            result.append(X_ordinal_raw)
        elif col in config['base_categories']:
            result.append(X_categorical_raw[0, categorical_idx])
            categorical_idx += 1
    
    return result

def generate_mask (X_tensor):
    '''
    Generate a mask for the changable columns (except categorical ones)
    '''
    config = get_data_config()

    changable_cols = config['optimizable_numerical_columns'] + config['optimizable_ordinal_columns']
    all_cols = config['columns']

    mask = torch.zeros_like(X_tensor)
    for col in changable_cols:
        mask[:, all_cols.index(col)] = 1
    return mask

def get_ordinal_feature_names():
    '''
    Get the names of the ordinal features.
    '''
    config = get_data_config()
    return config['all_ordinal_columns']

def get_categorical_to_optimize():
    '''
    Get the names of the categorical features to optimize.
    '''
    config = get_data_config()
    return config['optimizable_categories']

def get_recalc_params_matrix():
    '''
    Get the recalculation parameters matrix.
    '''
    config = get_data_config()
    from src.config_functions import ratio_func
    return [[
        config['recalc_params_matrix'][0][0],  # person_income
        config['recalc_params_matrix'][0][1],  # loan_amnt
        config['recalc_params_matrix'][0][2],  # loan_percent_income
        ratio_func  # the actual function
    ]]

def build_one_hot_groups():
    '''
    Build a dictionary of one-hot encoded feature groups from base categories
    '''
    config = get_data_config()
    base_categories = config['base_categories']
    all_cols = config['columns']
    one_hot_groups = {}
    for cat in base_categories:
        group = [col for col in all_cols if col.startswith(cat + "_")]
        if group:
            one_hot_groups[cat] = group
    return one_hot_groups

def filter_one_hot_groups(
    all_one_hot_groups: dict,
    allowed_group_names: list
) -> dict:
    """
    Filter a dictionary of all one-hot encoded feature groups to return only those allowed to change.

    Args:
        all_one_hot_groups (dict): Mapping from category name to list of feature indices.
                                   Example: {'loan_intent': [8, 9, 10, 11, 12, 13], ...}
        allowed_group_names (list): List of group names the user wants to change.

    Returns:
        dict: Filtered dict of only changable groups and their feature indices.
    """
    config = get_data_config()
    all_cols = config['columns']

    return {
        group: [all_cols.index(feature) for feature in features]
        for group, features in all_one_hot_groups.items()
        if group in allowed_group_names
    }

def form_recalc_dict(
        A_name, B_name, C_name, ratio_func, scaler):
    """
    Create a dictionary of parameters needed to recalculate a scaled feature C
    based on a relationship between two other scaled features A and B 
    (e.g., C = B / A in the original domain).

    Args:
        A_name (str): The column name for component A (e.g., 'person_income').
        B_name (str): The column name for component B (e.g., 'loan_amnt').
        C_name (str): The column name for component C (e.g., 'loan_percent_income').
        ratio_func (callable): A function ratio_func(a, b) -> float, defining how
            to compute the original-domain value of C from A and B.
        scaler (MinMaxScaler): A fitted MinMaxScaler that was trained on the 
            numeric columns, used for inverse transforming A/B and re-scaling C.

    Returns:
        dict: A dictionary containing the global/local indices for A, B, C,
              plus the calculation function and the scaler.
              This dictionary is meant to be passed to `recalculate_scaled()`.
    
    Notes:
        - Assumes you have global variables `all_cols` and `numerical_cols`
          which define the order of the full feature set and the numeric subset, 
          respectively.
    """
    config = get_data_config()
    all_cols = config['columns']
    numerical_cols = config['all_numerical_columns']

    idxA_global = all_cols.index(A_name)
    idxB_global = all_cols.index(B_name)
    idxC_global = all_cols.index(C_name)

    idxA_local = numerical_cols.index(A_name)
    idxB_local = numerical_cols.index(B_name)   
    idxC_local = numerical_cols.index(C_name)
    return {
        'global': [idxA_global, idxB_global, idxC_global],
        'local': [idxA_local, idxB_local, idxC_local],
        'calc_func': ratio_func,
        'scaler': scaler
        }

def get_ordinal_feature_idx(feature_name):
    '''
    Get the index of an ordinal feature in the full feature set.
    '''
    config = get_data_config()
    all_cols = config['columns']
    return all_cols.index(feature_name)
