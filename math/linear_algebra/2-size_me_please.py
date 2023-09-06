#!/usr/bin/env python3
"""
Este ejercicio permite obtener el shape de una matriz
"""

def matrix_shape(matrix):
    """
    Esta función retorna el shape de una matriz
    """
    column = len(matrix[0])
    row = len(matrix)
    if column > 2:
        layers = len(matrix[0][0])
        return [row, column, layers]
    return [row, column]
