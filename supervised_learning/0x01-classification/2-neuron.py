#!/usr/bin/env python3
"""
Neuron Forward Propagation
"""

Neuron = __import__("1-neuron").Neuron
import numpy as np

class Neuron(Neuron):
    """
    Class single Nueron performing binary classification
    """

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        nx, m = X.shape
        w = self.__W
        b = self.__b
        self.__A = np.matmul(w, X) + b
        self.A = self.__A

        return self.A
        
        
