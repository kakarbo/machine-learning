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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        num = 1
        d1 = nx
        for i in range(self.L):
            self.weights[f"W{num}"] = np.random.randn(layers[i], d1)
            self.weights[f"b{num}"] = np.zeros((layers[i], 1))
            num += 1
            d1 = layers[i]

