#!/usr/bin/env python3
"""
Este modulo concatena dos matrices
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Esta función retorna la concatenación de dos matrices
    """
    return np.concatenate((mat1, mat2), axis)
