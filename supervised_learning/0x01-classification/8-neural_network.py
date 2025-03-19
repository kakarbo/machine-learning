#!/usr/bin/env python3
"""
Neural Network
"""
import numpy as np


class NeuralNetwork:
    """
    Class NeuralNetwork
    """
    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if isinstance(nx, int):
            if nx < 1:
                raise ValueError("nx must be a positive integer")
        else:
            raise TypeError("nx must be an integer")

        if isinstance(nodes, int):
            if nodes < 1:
                raise ValueError("nodes must be a positive integer")
        else:
            raise TypeError("nodes must be an integer")
        self.W1 = np.random.randn(3, 784)
        self.b1 = np.zeros((3, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, 3)
        self.b2 = 0
        self.A2 = 0
