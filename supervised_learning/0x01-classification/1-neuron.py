#!/usr/bin/env python3
Neuron =  __import__('0-neuron').Neuron
import numpy as np
"""
Binary Classification
"""

class Neuron(Neuron):
    """
    Defined a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        class constructor
        """i
        super().__init__(nx)
        self.__W = self.W
        self.__b = 0
        self.__A = 0

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


