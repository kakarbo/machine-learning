#!/usr/bin/env python3
"""
Este modulo permite realizar transpose de una matriz 2D
"""

def matrix_transpose(matrix):
    """
    Esta función retorna una nueva matriz del resultado de transpose.
    """
    new_matrix = []
    num = 0
    for i in range(len(matrix[0])):
        vector = []
        for j in range(len(matrix)):
            vector.append(matrix[j][num])
        num += 1
        new_matrix.append(vector)
    

    return new_matrix
