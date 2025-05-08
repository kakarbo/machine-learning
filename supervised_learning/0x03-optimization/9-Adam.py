#!/usr/bin/env python3
"""
Adam
"""


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
    
