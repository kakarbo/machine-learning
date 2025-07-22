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
    cache = dict()
    cache[f'A0'] = X

    for i in range(L):
        Zi = np.matmul(
            weights[f'W{i+1}'], cache[f'A{i}']
        ) + weights[f'b{i+1}']
        if i == L - 1:
            cache[f'A{i+1}'] = np.exp(Zi) / (
                np.sum(np.exp(Zi), axis=0, keepdims=True)
            )
        else:
            cache[f'A{i+1}'] = np.tanh(Zi)
            boolean = np.random.rand(
                cache[f'A{i+1}'].shape[0],
                cache[f'A{i+1}'].shape[1]
            ) < keep_prob
            drop = np.where(boolean == 1, 1, 0)
            cache[f'A{i+1}'] *= drop
            cache[f'A{i+1}'] /= keep_prob
            cache[f'A{i+1}'] = drop
    return cache