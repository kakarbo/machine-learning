#!/usr/bin/env python3
"""
Gradient descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with dropout regularization using gradient
    descent.

    Args:
        Y: is a one-hot numpy.ndarray that contains the correct labels for the data.
        weights: Is a dictionary of the weights and biases of the neural network.
        cache: Is a dictionary of the outputs and dropout masks of each layer of the
        neural network.
        alpha: Is the learning rate.
        keep_prob: Is the probability that a node will be kept.
        L: is the number of layers of the network.
    
    Returns:
        cache: The weights of the network should be updated in place
    """
    a_dropout = list()
    D = list()
    for layer in range(L):
        W = weights[f'W{layer+1}']
        b = weights[f'b{layer+1}']
        A = cache[f'A{layer}']
        Z = np.matmul(W,A) + b

        if layer == L - 1:
            max = np.max(
                x, axis=1, keepdims=True
            )  # returns max of each row and keeps same dims
            e_x = np.exp(x - max)  # subtracts each row with its max value
            sum = np.sum(
                e_x, axis=1, keepdims=True
            )  # returns sum of each row and keeps same dims
            f_x = e_x / sum
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            a_dropout = (A*D) / keep_prob

        (a_dropout * D) / keep_prob
    return {}
    