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
        self.__W1 = np.random.randn(3, 784)
        self.__b1 = np.zeros((3, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, 3)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        matrix_product1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-matrix_product1))

        matrix_product2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-matrix_product2))

        return self.__A1, self.__A2

