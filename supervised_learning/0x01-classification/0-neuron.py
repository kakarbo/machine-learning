#!/usr/bin/env python3
"""
Classification
"""

class Neuron:
    """
    Defines a single neuron performing binary classification
    """
    w
    b = 0
    A = 0
    def __init__(self, nx):
        """
        class constructor
        """
        if isinstance(nx, int):
            if < 1:
                raise ValueError("nx must be a positive integer")

        else:
            raise TypeError("nx must be an integer")

