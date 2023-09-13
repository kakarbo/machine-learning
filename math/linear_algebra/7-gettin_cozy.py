#!/usr/bin/env python3
"""
Este modulo concatena dos matrices a lo largo de un eje especifico
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Funcion que concatena dos matrices
    """
    if axis == 0:
        return mat1 + mat2
    else:
        vector = []
        for i in range(len(mat1)):
            vector.append(mat1[i] + mat2[i])
        
        return vector

