#!/usr/bin/env python3
"""
Deep Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork
    """
    def __init__(self, nx, layers, activation="sig"):
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

        if activation is not "sig" or activation is not "tanh":
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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

    @property
    def activation():
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        e = 2.718
        self.__cache["A0"] = X
        for i in range(self.L):
            z = np.matmul(self.weights[f"W{i+1}"], self.cache[f"A{i}"]) + self.weights[f"b{i+1}"]
            if self.__activation == 'sig':
                self.__cache[f"A{i+1}"] = 1 / (1 + (np.exp(-z)))
            if self.__activation == 'tang':
                self.__cache[f"A{i+1}"] = ((np.exp(z)) - (np.exp(-z))) / ((np.exp(z)) + (np-exp(-z)))

        return A, self.cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        sigma = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(sigma))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.5):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        back = {}
        for i in range(self.L, 0, -1):
            A = cache[f"A{i - 1}"]
            if i == self.L:
                back[f"dz{i}"] = (cache[f"A{i}"] - Y)
            else:
                dz_prev = back[f"dz{i + 1}"]
                A_current = cache[f"A{i}"]
                back[f"dz{i}"] = (
                        np.matmul(W_prev.transpose(), dz_prev) *
                        (A_current * (1 - A_current)))
            dz = back[f"dz{i}"]
            dW = (1 / m) * (np.matmul(dz, A.transpose()))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            W_prev = self.weights[f"W{i}"]
            self.__weights[f"W{i}"] = (self.weights[f"W{i}"] - (alpha * dW))
            self.__weights[f"b{i}"] = (self.weights[f"b{i}"]) - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.5, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        """
        if isinstance(iterations, int):
            if iterations < 0:
                raise ValueError("iterations must be a positive integer")
        else:
            raise TypeError("iterations must be an integer")

        if isinstance(alpha, float):
            if alpha < 0:
                raise ValueError("alpha must be a positive")
        else:
            raise TypeError("alpha must be a float")

        if verbose or graph:
            if isinstance(step, int):
                if step < 0 and step <= iterations:
                    raise ValueError("step must be positive and <= iterations")
            else:
                raise TypeError("step musth be an integer")

        x_points = np.arange(0, iterations + 1, step)
        points = []
        for iteration in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and iteration % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {iteration} iterations: {cost}")
            if graph and iteration % step == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)

        iteration += 1
        if verbose and iteration % step == 0:
            cost = self.cost(Y, A)
            print(f"Cost after {iteration} iterations: {cost}")
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, "b")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.savefig("image/train-deep-neural-network-diagrama")
        prediction, cost = self.evaluate(X, Y)

        return prediction, cost

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Parameters:
            filename (file): is the file to which the
            object should be saved
        """
        if not filename.endswith("pkl"):
            filename = f"{filename}.pkl"

        file_obj = open(filename, "wb")
        pickle.dump(self, file_obj)
        file_obj.close()

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Parameters:
            filename (file): is the file from which the
            object shoul be loaded

        Returns
            filename: the loaded object, or None if
            filename doesn't exist
        """
        try:
            file_obj = open(filename, "rb")
            load_obj = pickle.load(file_obj)
            file_obj.close()

            return load_obj
        except FileNotFoundError:
            return None
