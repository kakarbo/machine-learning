#!/usr/bin/env python3
"""
Precision
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Parameters:
        cosfusion (numpy.ndarray): is a confusion of shape (classes, classes) where row
        indices represent the correct labels and column indices represent the predicted
        labels
            - classes (int): is the number of classes
    
    Returns:
        precision (numpy.ndarray): of shape (classes,) containing the precision of each class
    """
    precisiones = np.zeros(confusion.shape[0])
    

