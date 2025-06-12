#!/usr/bin/env python3
"""
Gradient descent with l2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent with L2
    regularization.

    Parameters:
        Y (numpy.ndarray): of shape(classes, m) that contains the correct labels for
        the data.
        weights (dict): is a dictionary of the weights and biases of the neural network.
        cache (dict): is a dictionary of each layer of the neural network.
        alpha (float): is the learning rate.
        lambtha (float): is the L2 regularization parameter.
        L (int): is the number of layers of the network.
    
    Returns:
        the weights and biases of the network should be updated in place.
    """