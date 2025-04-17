import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dotenv import load_dotenv
load_dotenv()

from src.scaler import get_scaler
from src.model import get_model
from src.data import generate_mask, build_one_hot_groups, filter_one_hot_groups, form_recalc_dict, get_ordinal_feature_names, get_categorical_to_optimize, get_recalc_params_matrix, get_ordinal_feature_idx, transform, inverse_transform
from src.utils import recalculate_scaled, soft_round, init_logits_from_original, gumbel_softmax_sample

import yaml

config_path = f'configs/counterfactual_explanations.yaml'

with open(config_path, 'r') as file:
        config =  yaml.safe_load(file)



def generate_counterfactual(
        model,
        x_original,
        mask,                                # a mask for the gradients to fix the non changable features
        scaler,                              # scaler we used to scale original data during processing
        recalc_params_matrix,                # matrix of feature relations needed recalculation 
        ordinal_feature_names,               # a list of ordinal feature names
        one_hot_feature_groups_to_optimize,  # dict of one-hot groups we want to change
        target_label=0,                      # 0 for "non-default", 1 for "default"
        lambda_param=1.0,                    # trade-off coefficient between distance & classification loss
        alpha_param=10.0,                    # weight for integer penalty
        lr=0.01,                             # learning rate for gradient descent
        max_steps=500,                       # max optimization steps
        temperature=0.5,                     # Gumbel-softmax temperature
        distance_metric='l2' ):
    
    """
    Generate a counterfactual while handling categorical (one-hot) features using Gumbel-Softmax.

    - Applies Gumbel-Softmax to allow differentiable optimization over one-hot groups.
    - Respects gradient masking for frozen features.
    - Minimizes BCE loss to push the model toward a target label.
    - Adds a penalty to keep ordinal features near integer values.
    - After optimization, snaps both categorical and ordinal features to valid final values.

    Args:
        model (nn.Module): A trained model accepting scaled inputs.
        x_original (torch.Tensor): Shape [1, num_features], original scaled data point.
        mask (torch.Tensor): Shape [1, num_features], 1 = allowed to change, 0 = frozen.
        scaler (MinMaxScaler): Scaler for inverse transforms (relationship recalculations).
        recalc_params_matrix (list): Each item is [A, B, C, func], for recomputing related features.
        ordinal_feature_names (list): Names of ordinal features.
        one_hot_feature_groups_to_optimize (dict): Group name → list of one-hot column indices.
        target_label (int): Desired label for counterfactual (0 or 1).
        lambda_param (float): Balances distance vs. classification loss.
        alpha_param (float): Penalty weight for ordinal values drifting from integers.
        lr (float): Learning rate for optimizer.
        max_steps (int): Number of gradient steps.
        temperature (float): Temperature for Gumbel-Softmax distribution.
        distance_metric (str): 'l2' or 'l1' to define the distance measure.
    Returns:
        torch.Tensor: The final counterfactual, shape [1, num_features], on CPU,
                      with categorical groups snapped to valid one-hot, and ordinals rounded.
    """

    x_original = x_original.to(model.device).detach()
    x_cf = x_original.clone().requires_grad_(True)
    mask = mask.to(model.device)

    logits_dict = {
        # group: init_logits_from_original(x_original, indices)
        group: torch.nn.Parameter(torch.randn(len(indices)).to(model.device))
        for group, indices in one_hot_feature_groups_to_optimize.items()
    }

    optimizer = Adam([x_cf, *logits_dict.values()], lr=lr)

    for step in range(max_steps):
        optimizer.zero_grad()

        # Insert Gumbel-softmax outputs into x_cf
        for group, logits in logits_dict.items():
            indices = one_hot_feature_groups_to_optimize[group]
            probs = gumbel_softmax_sample(logits, temperature)
            with torch.no_grad():
                x_cf[0, indices] = probs

        # Calculate distance: encourage counterfactual to be close to original
        if distance_metric == 'l2':
            distance = torch.norm(x_cf - x_original, p=2)
        else : # use 'l1'
            distance = torch.norm(x_cf - x_original, p=1)
        
        # Calculate classification loss of the cf from the model prediction
        logits = model.forward(x_cf)
        label_tensor = torch.tensor([float(target_label)]).to(x_cf.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, label_tensor)

        penalty = .0
        for ord in ordinal_feature_names:
            idx = get_ordinal_feature_idx(ord)
            penalty += torch.abs(x_cf[0, idx] - soft_round(x_cf[0, idx], alpha=50.0))
        penalty *= alpha_param

        #Calculate main formula of the method with added ordinal feature tweeking penalty
        loss = distance + lambda_param * bce_loss + penalty

        loss.backward()

        # Zero out gradients on frozen features
        with torch.no_grad():
            x_cf.grad *= mask
        
        optimizer.step()

        # Recompute relationships after the optimization step
        with torch.no_grad():
            # Iterate through relationship matrix to do the recalculation
            for rel in recalc_params_matrix:
                A, B, C, func = rel
                param_dict = form_recalc_dict(A, B, C, func, scaler)
                x_cf_updated = recalculate_scaled(x_cf, param_dict)
                x_cf.copy_(x_cf_updated)

    # do the final snapping of ordinal features
    with torch.no_grad():
        for group, logits in logits_dict.items():
            indices = one_hot_feature_groups_to_optimize[group]
            hard = torch.zeros_like(logits)
            hard[torch.argmax(logits)] = 1.0
            x_cf[0, indices] = hard

        for ord in ordinal_feature_names:
            idx = get_ordinal_feature_idx(ord)
            x_cf[0, idx] = torch.round(x_cf[0, idx])

    return x_cf.detach().cpu()


def counterfactual_explanation(
    X_original,
    y_original):
    """
    Generate a counterfactual explanation for a given data point.
    NOTE that X_original is not scaled, it is the original data point(dictionary).
    """

    X_original = transform(X_original)
    scaler = get_scaler()
    model, model_device = get_model()

    mask = generate_mask(X_original)
    ordinal_feature_names = get_ordinal_feature_names()
    all_one_hot_groups = build_one_hot_groups()
    categorical_to_optimize = get_categorical_to_optimize()
    one_hot_groups_to_optimize = filter_one_hot_groups(all_one_hot_groups, categorical_to_optimize)
    recalc_params_matrix = get_recalc_params_matrix()

    target_label = 1 if y_original == 0 else 0
    x_cf = generate_counterfactual(
        model=model,
        x_original=X_original,
        mask=mask,
        scaler=scaler,
        recalc_params_matrix=recalc_params_matrix,
        ordinal_feature_names=ordinal_feature_names,
        one_hot_feature_groups_to_optimize=one_hot_groups_to_optimize,
        target_label=target_label,
        lambda_param=config['lambda_param'],
        alpha_param=config['alpha_param'],
        lr=config['lr'],
        max_steps=config['max_steps'],
        temperature=config['temperature'],
        distance_metric=config['distance_metric']
    )
    return inverse_transform(x_cf)
    

