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
    weights_copy = weights.copy()
    for i in range(L, 0, -1):
        m = Y.shape[1]
        if i != L:
            # all layers use a tanh activation, except last
            # introduce call to tanh_prime method
            dZi = np.multiply(np.matmul(
                weights_copy['W' + str(i + 1)].T, dZi
            ), tanh_prime(cache['A' + str(i)]))
            # pass dZi through same dropout mask as that
            # saved in cache during forward_prop
            # dropout mask applied to hidden layers only
            # regularize and normalize by keep_prob
            dZi *= cache['D' + str(i)]
            dZi /= keep_prob
        else:
            # last layer uses a softmax activation
            dZi = cache['A' + str(i)] - Y
        dWi = np.matmul(dZi, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(dZi, axis=1, keepdims=True) / m
        weights['W' + str(i)] = weights_copy['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = weights_copy['b' + str(i)] - alpha * dbi


def tanh_prime(Y):
    """define the derivative of the activation function tanh"""
    return 1 - Y ** 2
    