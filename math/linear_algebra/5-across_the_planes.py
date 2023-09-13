#!/usr/bin/env python3
"""
Este modulo suma dos matrices 2D de forma elemental
"""


def add_matrices2D(mat1, mat2):
    """
    Función que suma dos matrices de forma elemental
    """
    if len(mat1[0]) != len(mat2[0]):
        return None
    
    new_matrix = []
    for i in range(len(mat1)):
        vector = []
        for j in range(len(mat2)):
            vector.append(mat1[i][j] + mat2[i][j])
        new_matrix.append(vector)

    return new_matrix
