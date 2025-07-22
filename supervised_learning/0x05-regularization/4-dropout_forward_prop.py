#!/usr/bin/env python3
"""
Forward Propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: is a numpy.ndarray that containing the input data for network.
        weihgts: Is a dictionary of the weights and biases of the neural network.
        L: the number of layers in the network.
        keep_prob: Is the probability that a node will be kept.

    Returns:
        A dictionary containing the outputs of each layer and the dropout mask used
        on each layer.
    """
    cache = {"A0": X}

    for layer in range(L):
        W = weights["W" + str(layer + 1)]
        b = weights["b" + str(layer + 1)]
        A = cache["A" + str(layer)]
        Z = np.matmul(W, A) + b

        if layer == L - 1:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A *= D
            A /= keep_prob
            cache["D" + str(layer + 1)] = D

        cache["A" + str(layer + 1)] = A
    return cache