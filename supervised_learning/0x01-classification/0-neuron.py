#!/usr/bin/env python3

import numpy as np
"""
Classification
"""

class Neuron:
    """
    Defines a single neuron performing binary classification
    """
    W = np.random.randn(1, 784)
    b = 0
    A = 0
    def __init__(self, nx):
        """
        class constructor
        """
        if isinstance(nx, int):
            if nx < 1:
                raise ValueError("nx must be a positive integer")

        else:
            raise TypeError("nx must be an integer")

