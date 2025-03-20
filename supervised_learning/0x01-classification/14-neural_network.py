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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        sigma_hidden_layer = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(sigma_hidden_layer))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network´s predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        d__W2 = (1 / m) * np.matmul(dz2, A1.transpose())
        d__b2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.W2.transpose(), dz2) * (A1 * (1 - A1))
        d__W1 = (1 / m) * np.matmul(dz1, X.transpose())
        d__b1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)
        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        """
        if isinstance(iterations, int):
            if iterations < 0:
                raise ValueError("iterations must be a positive integer")
        else:
            raise TypeError("iterations must be an integer")

        if isinstance(alpha, float):
            if alpha < 0:
                raise ValueError("alpha must be positive")
        else:
            raise TypeError("alpha must be a float")

        for iteration in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        prediction, cost = self.evaluate(X, Y)

        return prediction, cost

