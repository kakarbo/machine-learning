#!/usr/bin/env python3
"""
"""
import numpy as np


def cat_matrices(mat1, mat2, axis=0):
    """
    """
    if isinstance(mat1[axis], list) and isinstance(mat2[axis], list):
        print("INGRESA")
        row_mat1 = len(mat1)
        column_mat1 = len(mat1[0])

        row_mat2 = len(mat2)
        column_mat2 = len(mat2[0])

        if (row_mat1, column_mat1) != (row_mat2, column_mat2):
            print("INGRESA PARA SALIR")
            return None


    return np.concatenate((mat1, mat2), axis)
