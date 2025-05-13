#!/usr/bin/env python3
"""
Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Parameters:
        Z (numpy.ndarray): of shape (m, n) that should be normalized
            m (int): is the number of data points.
            n (int): is the number of features in Z.
        gamma (numpy.ndarray): of shape (1, n) containing the scales used for
        batch normalization
        beta (numpy.ndarray): of shape (1, n) containing the offsets used for
        batch normalization
        epsilon (float): is a small number used to avoid divsion by zero
    
    Returns:
        numpy.ndarray: The normalized Z matrix
    """
    # Paso 1: Calcular la media y varianza pof feature (axis=0)
    mu = np.mean(Z, axis=0, keepdims=True)
    sigma2 = np.var(Z, axis=0, keepdims=True)

    # Paso 2: Normalizar Z
    Z_norm = (Z - mu) / np.sqrt(sigma2 + epsilon)

    # Paso 3: Aplicar escala (gamma) y desplazamiento (beta)
    Z_tilde = gamma * Z_norm + beta

    return Z_tilde
