#!/usr/bin/env python3
"""
Este modulo permite sumar dos matrices que tiene igual shape
"""
import numpy as np


def add_matrices(mat1, mat2):
    """
    Esta función retorna la suma de dos matrices
    """
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        if len(mat1[0]) != len(mat2[0]):
            return None
    else:
        if len(mat1) != len(mat2):
            return None
    
    return np.add(mat1, mat2)
