#!/usr/bin/env python3
"""
Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Parameters:
        cosfusion (numpy.ndarray): is a confusion of shape (classes, classes) where row
        indices represent the correct labels and column indices represent the predicted
        labels
            - classes (int): is the number of classes
    
    Returns:
        sensitivity (numpy.ndarray): of shape (classes,) containing the sensitivity of
        each class
    """
    sensitivities = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        TP = confusion[i, i]
        FN = np.sum(confusion[i, :]) - TP

        sensitivities[i] = TP / (TP + FN) if (TP + FN) != 0 else 0.0

    return sensitivities
