#!/usr/bin/env python3
"""
Normalize
"""


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix:

    Parameters:
    X (numpy.ndarray): of shape (d, nx) to normalize
        d (int): is the number of data points
        nx (int): is the number of features
    m (numpy.ndarray): of shape (nx,) that contains the mean of all features
    of X
    s (numpy.ndarray): of shape (nx,) that contains the standard deviation of
    all features of X

    Returns:
        The normalized X matrix
    """
    X_norm = (X - m) / s
    return X_norm
