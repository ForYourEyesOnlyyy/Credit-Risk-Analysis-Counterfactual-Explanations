'''
This file contains utility functions for the counterfactual explanations.
- recalculate_scaled
- soft_round
- init_logits_from_original
- gumbel_softmax_sample
'''


import torch
import torch.nn.functional as F


def recalculate_scaled(
    x_scaled_full: torch.Tensor,
    recacl_param_dict: dict
) -> torch.Tensor:
    """
    Update one scaled feature (C) in x_scaled_full based on a user-defined
    relationship involving two other scaled features (A and B). Only the 
    feature C column is changed; all other values remain identical.

    Args:
        x_scaled_full (torch.Tensor): A tensor of shape [1, num_features],
            containing scaled values for the entire feature vector.
        recacl_param_dict (dict): A dictionary returned by `form_recalc_dict()`, 
            containing:
            - 'global': list of [idxA_global, idxB_global, idxC_global] 
              for the feature indices in the full input.
            - 'local': list of [idxA_local, idxB_local, idxC_local] 
              for their indices in the numeric array used by the scaler.
            - 'calc_func': a function calc_func(a, b) -> c in the original domain.
            - 'scaler': the fitted MinMaxScaler used to do partial inverse.
    
    Returns:
        torch.Tensor: A new tensor of shape [1, num_features], where only
            the feature C column has been recalculated and re-scaled 
            based on the relationship between A and B in the original domain.

    Notes:
        - The function does not alter A or B themselves, nor any other columns.
        - This approach ensures only the target column C is re-scaled.
        - The user can define any function for `calc_func(a, b)` to represent 
          the original-domain relationship (e.g., c = b / a).
    """
    device = x_scaled_full.device
    dtype  = x_scaled_full.dtype

    # Unpack recalculation params
    idxA_global, idxB_global, idxC_global = recacl_param_dict['global']
    idxA_local, idxB_local, idxC_local = recacl_param_dict['local']
    calc_func = recacl_param_dict['calc_func']
    scaler = recacl_param_dict['scaler']
    
    x_cpu = x_scaled_full.clone().detach().cpu().numpy()
    
    scaledA = x_cpu[0, idxA_global]
    scaledB = x_cpu[0, idxB_global]
    
    # Retrieve the min/max from scaler for local indices
    minA = scaler.data_min_[idxA_local]
    maxA = scaler.data_max_[idxA_local]
    minB = scaler.data_min_[idxB_local]
    maxB = scaler.data_max_[idxB_local]
    minC = scaler.data_min_[idxC_local]
    maxC = scaler.data_max_[idxC_local]

    # Convert scaled -> original domain
    #    orig_val = scaled_val * (max - min) + min
    origA = scaledA * (maxA - minA) + minA
    origB = scaledB * (maxB - minB) + minB

    # Compute new C in the original domain
    origC = calc_func(origA, origB)

    # Re-scale that new C to [0,1]
    #    scaledC = (origC - minC) / (maxC - minC)
    denomC = (maxC - minC)
    if denomC == 0:
        scaledC = 0.0
    else:
        scaledC = (origC - minC) / denomC
    # optional clamp
    scaledC = max(0.0, min(1.0, scaledC))

    # Inject new scaledC back
    x_cpu[0, idxC_global] = scaledC

    x_updated = torch.tensor(x_cpu, device=device, dtype=dtype)
    return x_updated


def soft_round(x, alpha= 10.0):
    """
    Differentiable approximation to round(x).
    It keeps the integer part floor(x),
    and replaces the fractional part with a smooth function that transitions around 0.5.

    Args:
      x: A PyTorch Tensor (e.g. shape [1]).
      alpha: Controls sharpness. Higher alpha ~ sharper transition near 0.5.

    Returns:
      A "softly rounded" version of x, still in float, but
      fraction is replaced by a sigmoid-based approximation around 0.5.
    """
    x_int = torch.floor(x)
    x_floating = x - x_int

    # Sigmoid-based fraction, so that at x_floating=0.5 we transition
    x_floating_approx = torch.sigmoid(alpha * (x_floating - 0.5))

    return x + x_floating_approx


def init_logits_from_original(x_original, group_indices):
    '''
    Initialize logits from the original values.
    Args:
        x_original (torch.Tensor): The original values, shape [1, num_features]
        group_indices (list): The indices of the group to initialize
    Returns:
        torch.nn.Parameter: The initialized logits
    '''
    probs = x_original[0, group_indices].clone()
    logits = torch.log(probs + 1e-6)  # avoid log(0)
    return torch.nn.Parameter(logits.to(x_original.device))

def gumbel_softmax_sample(logits, temperature=0.5):
    """
    Samples a soft one-hot vector using the Gumbel-Softmax trick.

    Args:
        logits (torch.Tensor): Unnormalized log-probabilities, shape [num_categories]
        temperature (float): Controls the sharpness of the softmax

    Returns:
        torch.Tensor: Soft one-hot vector of shape [num_categories]
    """
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return F.softmax((logits + gumbel_noise) / temperature, dim=-1)