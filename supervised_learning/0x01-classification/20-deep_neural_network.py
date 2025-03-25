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
        num = 1
        d1 = nx
        for i in range(self.__L):
            self.__weights[f"W{num}"] = np.random.randn(layers[i], d1)
            self.__weights[f"b{num}"] = np.zeros((layers[i], 1))
            num += 1
            d1 = layers[i]

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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        sigma = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 * A)))
        cost = (1 / m) * (-(sigma))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A1)
        prediction = np.where(A1 >= 0.5, 1, 0)

        return prediction, cost
