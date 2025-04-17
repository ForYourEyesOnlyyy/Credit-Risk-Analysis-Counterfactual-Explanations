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
from src.scaler import get_scaler
from src.encoder import get_encoder
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

data_config_path = 'configs/data.yaml'

with open(data_config_path, 'r') as file:
    config = yaml.safe_load(file)

def transform(raw_input: dict) -> torch.Tensor:
    """
    Transform a single raw input dictionary into a model-ready tensor.

    Args:
        raw_input (dict): A dict with keys from config['raw_columns']

    Returns:
        torch.Tensor: A tensor ready to be passed to the model
    """
    # Load preprocessing objects
    scaler = get_scaler()  # Assumes a StandardScaler or similar
    onehot_encoder = get_encoder()  # Assumes a fitted OneHotEncoder

    # Step 1: Create DataFrame from dict input
    df = pd.DataFrame([raw_input], columns=config['raw_columns'])

    # Step 2: Scale numerical columns
    numerical_cols = config['all_numerical_columns']
    X_numerical = df[numerical_cols]
    X_numerical_scaled = scaler.transform(X_numerical)
    X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_cols)

    # Step 3: Encode ordinal columns
    ordinal_mapping = {'OTHER': 0, 'RENT': 1, 'MORTGAGE': 2, 'OWN': 3}
    X_ordinal = df[config['all_ordinal_columns']].replace(ordinal_mapping).astype(float)

    # Step 4: One-hot encode categorical columns
    categorical_df = df[config['base_categories']]
    X_categorical_encoded = onehot_encoder.transform(categorical_df).toarray()
    onehot_cols = onehot_encoder.get_feature_names_out(config['base_categories'])
    X_categorical_encoded = pd.DataFrame(X_categorical_encoded, columns=onehot_cols)

    # Step 5: Combine everything
    X_all = pd.concat(
        [X_numerical_scaled.reset_index(drop=True),
         X_ordinal.reset_index(drop=True),
         X_categorical_encoded.reset_index(drop=True)],
        axis=1
    )

    # Step 6: Reorder columns to match final model input order
    missing_cols = [col for col in config['columns'] if col not in X_all.columns]
    for col in missing_cols:
        X_all[col] = 0  # Add missing one-hot columns as zeros

    X_all = X_all[config['columns']]  # Ensure correct order

    # Convert to tensor
    X_tensor = torch.tensor(X_all.values, dtype=torch.float)

    return X_tensor



def inverse_transform(X_tensor: torch.Tensor) -> dict:
    """
    Inverse transforms a model-ready tensor back to raw input format.

    Args:
        X_tensor (torch.Tensor): Tensor with shape (1, n_features) in model input format.

    Returns:
        dict: Dictionary with keys matching config['raw_columns'].
    """
    scaler = get_scaler()
    encoder = get_encoder()

    # Convert tensor to DataFrame
    X = X_tensor.detach().cpu().numpy()
    df_transformed = pd.DataFrame(X, columns=config['columns'])

    # Inverse numerical scaling
    numerical_cols = config['all_numerical_columns']
    X_numerical_scaled = df_transformed[numerical_cols]
    X_numerical = scaler.inverse_transform(X_numerical_scaled)
    X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_cols)

    # Decode ordinal column(s)
    ordinal_mapping = {0: 'OTHER', 1: 'RENT', 2: 'MORTGAGE', 3: 'OWN'}
    ordinal_vals = {}
    for col in config['all_ordinal_columns']:
        raw_val = int(round(df_transformed[col].values[0]))
        ordinal_vals[col] = ordinal_mapping.get(raw_val, 'UNKNOWN')

    # Inverse one-hot encoding
    # Extract only one-hot encoded part
    onehot_cols = encoder.get_feature_names_out(config['base_categories'])
    X_onehot = df_transformed[onehot_cols].values
    X_categorical_decoded = encoder.inverse_transform(X_onehot)

    # Create dict for categorical decoded values
    categorical_vals = dict(zip(config['base_categories'], X_categorical_decoded[0]))

    # Combine all into final result dict
    result = {}
    for col in config['raw_columns']:
        if col in numerical_cols:
            val = X_numerical_df[col].values[0]
            # Format nicely
            if col in ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']:
                result[col] = int(round(val))
            else:
                result[col] = float(f"{val:.2f}")
        elif col in config['all_ordinal_columns']:
            result[col] = ordinal_vals[col]
        elif col in config['base_categories']:
            result[col] = categorical_vals[col]

    return result

def generate_mask (X_tensor):
    '''
    Generate a mask for the changable columns (except categorical ones)
    '''

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
    return config['all_ordinal_columns']

def get_categorical_to_optimize():
    '''
    Get the names of the categorical features to optimize.
    '''
    return config['optimizable_categories']

def get_recalc_params_matrix():
    '''
    Get the recalculation parameters matrix.
    '''
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
    all_cols = config['columns']
    return all_cols.index(feature_name)
