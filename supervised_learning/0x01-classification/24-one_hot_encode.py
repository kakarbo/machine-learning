#!/usr/bin/env python3
"""
One-Hot encode
"""
import numpy as np

def one_hot_encode(Y, classes):
    """
    Function one hot encode
    Converts a numeric label vector into a one-hot matrix

    Parameters:
        Y (numpy.ndarray): Containing numeric class labels.
        classes (int): is the maximum number of classes 
        found in Y.

    Returns:
        numpy.ndarray: a one-hot encoding of Y with 
        shape(classes, m), or None on Failure
    """
    one_hot_encode = np.empty((10, 10))
    for num_compare in range(classes):
        one_hot = np.zeros((classes,), dtype=int)
        for value in range(classes):
            if num_compare == Y[value]:
                one_hot[value] = 1
        one_hot_encode[num_compare] = one_hot

    return one_hot_encode
        



