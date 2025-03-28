#!/usr/bin/env python3

import numpy as np
"""
Classification
"""

class Neuron:
    """
    Defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        class constructor
        """
        if isinstance(nx, int):
            if nx < 1:
                raise ValueError("nx must be a positive integer")
            self.W = np.random.randn(1, 784)
            self.b = 0
            self.A = 0
        else:
            raise TypeError("nx must be an integer")

