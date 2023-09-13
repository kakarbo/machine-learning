#!/usr/bin/env python2
"""
Este modulo permite realizar operaciones matematicas con numpy
"""
import numpy as np


def np_elementwise(mat1, mat2):
    """
    Esta función retorna las diferentes operaciones matematicas como:
    * suma
    * resta
    * multiplicación
    * división
    """
    return (
        np.add(mat1, mat2),
        np.subtract(mat1, mat2),
        np.multiply(mat1, mat2),
        np.divide(mat1, mat2)
    )
