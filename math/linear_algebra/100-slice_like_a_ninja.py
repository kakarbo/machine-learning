#!/usr/bin/env python3
"""
Este modulo permite realiza un corte en la matriz con tuplas
"""


def np_slice(matrix, axes={}):
    """
    Esta función retorna el corte de la matriz
    """
    if len(axes) > 1:
        slice_tupla = axes[0]
        num1, = slice_tupla
        slice_tupla = axes[2]
        _, _, num2 = slice_tupla
        return matrix[:num1, :, ::num2]
    else:
        slice_tupla = axes[1] 
        num1, num2 = slice_tupla
        return matrix[:, num1:num2]
