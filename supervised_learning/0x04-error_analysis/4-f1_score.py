#!/usr/bin/env python3
"""
F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix.

    Parameters:
        confusion (numpy.ndarray): of shape(classes, classes) where row indices represent
        the correct labels and column indices represent the predicted labels

    Returns:
        confusion (numpy.ndarray): of shape(classes,) containing the F1 score of each classs
    """
    recall = sensitivity(confusion)
    precisions = precision(confusion)
    f1Score = np.zeros(recall.shape[0])
    for i in range(precisions.shape[0]):
        f1Score[i] = 2  * (precisions[i] * recall[i]) / (precisions[i] + recall[i])
    return f1Score
