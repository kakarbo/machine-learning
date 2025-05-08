#!/usr/bin/env python3
"""
RMSProp
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
        alpha (float): is the learning rate
        beta2 (): is the RMSProp weight
        epsilon (): is a small number to avoid division by zero
        var (numpy.ndarray): containing the variable to be updated
        grad (numpy.ndarray): containing the gradient of var
        s (numpy.ndarray): is the previous second moment of var
    
    Returns:
        (Tuple) - The updated variable and the new moment, respectively
    """
    updated_s = beta2 * s + (1 - beta2) * (grad ** 2)
    updated_var = var - alpha * (grad / (np.sqrt(updated_s) + epsilon))

    return updated_var, updated_s
