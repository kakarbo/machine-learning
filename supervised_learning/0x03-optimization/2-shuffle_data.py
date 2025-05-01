#!/usr/bin/env python3
"""
Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Parameters:
        X (numpy.ndarray): of shape(m, nx) to shuffle
            m (int): is the number of data points
            nx (int): is the number of features in x
        Y (numpy.ndarray): of shape(m, ny) to shuffle
            m (int): is the same number of data points as in X
            ny (int): is the number of features in Y

    Returns:
        the suffled X and Y matrices
    """
    m, nx = X.shape
    indices = np.random.permutation(m)

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    return X_shuffled, Y_shuffled
