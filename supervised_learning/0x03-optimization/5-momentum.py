#!/usr/bin/env python3
"""
Momentum
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    algorithm.

    Parameters:
        alpha (float): is the learning rate.
        beta1 (float): is the momentum weight
        var (numpy.ndarray): containing the variable to be updated
        grad (numpy.ndarray): containing the gradient of var
        v (numpy.ndarray): is the previous first moment of var
    
    Returns:
        tuple: (var_updated, v_updated) - The updated variable and the new moment, respectively
    """
    v_updated  = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_updated

    return var_updated, v_updated
