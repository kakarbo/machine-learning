#!/usr/bin/env python3
"""
One-Hot Decode
"""
import numpy as np

def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels

    Parameters:
        one_hot (numpy.ndarray): is a one-hot encoded

    Returns:
        numpy.ndarray: containing the numeric labels for
        each example, or None on failure
    """
    classes, m = one_hot.shape
    if len(one_hot) <= 0:
        return None
    one_hot_decode = np.zeros((classes,), dtype=int)
    for num_classes in range(classes):
        for num_examples in range(m):
            if 1 == one_hot[num_classes, num_examples]:
                one_hot_decode[num_examples] = num_classes

    return one_hot_decode
            

