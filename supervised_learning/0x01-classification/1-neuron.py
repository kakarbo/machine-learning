#!/usr/bin/env python3

import numpy as np
"""
Binary Classification
"""

class Neuron:
    """
    Defined a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        class constructor
        """
        if isinstance(nx, int):
            if nx < 1:
                raise ValueError("nx must be a positive integer")
            self.__W = np.random.randn(1, 784)
            self.__b = 0
            self.__A = 0
        else:
            raise TypeError("nx must be an integer")

    @property
    def W(self):
        """
        Returns: private instance weight
        """
        return self.__W

    @property
    def b(self):
        """
        Returns: private instance bias
        """
        return self.__b

    @property
    def A(self):
        """
        Returns: private instace output
        """
        return self.__A


