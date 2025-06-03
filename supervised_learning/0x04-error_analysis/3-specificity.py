#!/usr/bin/env python3
"""
Specificity
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix:

    Parameters:
        confusion (numpy.ndarray): of shape (classes, classes) where row indices
        represent the correct labels and column indices represent the predicted labels
    
    Returns:
        confusion (numpy.ndarray): of shape (classes,) containing the specificity
        of each class
    """
    specificity = np.zeros(confusion.shape[0])
    print(confusion)
