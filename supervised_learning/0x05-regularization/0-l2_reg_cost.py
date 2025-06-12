#!/usr/bin/env python3
"""
L2 REgularization cost
"""
import numpy as np
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
        cost (numpy.ndarray): is the cost of the network without L2 regularization.
        lambtha (float): es the regularization parameter.
        weights (dict): is a dictionary of the weights and biases(numpy.ndarray) of  the
        neural network.
        L (int): is the number of layers in the neural network.
        m (int): is the number of data points used.
    
    Returns:
        the cost of the network accounting for L2 regularization
    """
    L2 = 0
    for layer in range(L):
        L2 += np.linalg.norm(weights[f'W{layer + 1}'])
        L2 += lambtha / 2 / m * zigma
    cost = cost + lambtha / (2 * m) * L2
    return cost
