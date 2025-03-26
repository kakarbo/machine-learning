#!/usr/bin/env python3
"""
Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork
    """
    def __init__(self, nx, layers):
        """
        class constructor
        """
        if isinstance(nx, int):
            if nx < 1:
                raise ValueError("nx must be a positive integer")
        else:
            raise TypeError("nx must be an integer")
        if isinstance(layers, list) or len(list) == 0:
            copy_layers = layers.copy()
            copy_layers.sort()
            if copy_layers[0] < 0:
                raise TypeError("layers must be a list of positive integers")
        else:
            raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        previous = nx
        for i, layer in enumerate(layers, 1):
            self.weights[f"W{i}"] = (np.random.randn(layer, previous) * np.sqrt(2 / previous))
            self.weights[f"b{i}"] = np.zeros((layer, 1))
            previous = layer

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for i in range(self.L):
            z = np.matmul(self.weights[f"W{i+1}"], self.cache[f"A{i}"]) + self.weights[f"b{i+1}"]
            self.__cache[f"A{i+1}"] = 1 / (1 + np.exp(-z))
        return self.cache[f"A{i+1}"], self.cache

