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
    m = Y.shape[1]
    grads = {}

    # layer last
    A_L = cache[f"A{L}"]
    dz = A_L - Y

    for layer in reversed(range(1, L + 1)):
        A_prev = cache[f'A{layer-1}'] if layer > 1 else cache['A0']
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        # Gradients
        dW = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # Update
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dz)
            dz = dA_prev * (1 - A_prev ** 2)
