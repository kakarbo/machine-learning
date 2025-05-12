#!/usr/bin/env python3
"""
Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Parameters:
        alpha (float): is the learning rate
        beta1 (float): is the weight used for the first moment
        beta2 (float): is the weight used for the second moment
        epsilon (float): is a small number to avoid division by zero
        var (numpy.ndarray): containing the variable to be updated
        grad (numpy.ndarray): containing the gradient of var
        v (numpy.ndarray): is the previous first moment of var
        s (numpy.ndarray): is the previous second moment of var
        t (int): is the time step used for bias correction
    
    Returns
        (Tuple): the updated variable, the new first moment, and the new second
        moment, respectively
    """
    # First step: update the moment (momentum and RMSProp)
    updated_v = beta1 * v + (1 - beta1) * grad
    updated_s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Second step: correction of bias
    v_corrected = updated_v / (1 - beta1 ** t)
    s_corrected = updated_s / (1 - beta2 ** t)

    # Three step: update variable
    updated_var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return updated_var, updated_v, updated_s
