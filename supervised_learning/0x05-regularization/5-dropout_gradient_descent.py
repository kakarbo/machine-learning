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
    