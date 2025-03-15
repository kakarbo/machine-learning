#!/usr/bin/env python3
"""
Neuron Forward Propagation
"""
import numpy as np


class Neuron:
    """
    Class single Nueron performing binary classification
    """
    def __init__(self, nx):
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        """
        nx, m = X.shape
        w = self.__W
        b = self.__b
        matrix_product = np.matmul(w, X) + b
        sigmoid = 1 / (1 + np.exp(-matrix_product))
        self.__A = sigmoid

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        """
        m = Y.shape[1]
        dz = A - Y
        d_W = (1 / m) * np.matmul(X, dz.transpose()).transpose()
        d_b = (1 / m) * np.sum(dz)
        self.__W = self.W - (alpha * d_W)
        self.__b = self.b - (alpha * d_b)



