#!/usr/bin/env python3
"""
Mini-Batch
"""


def train_mini_batch(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    batch_size=32,
    epochs=5,
    load_path="/tmp/model.ckpt",
    save_path="/tmp/model.ckpt"
):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    Parameters:
        X_train (numpy.ndarray): of shape(m, 784) containing the training data
        Y_train (numpy.ndarray): of shape(m, 10) containing the training labels
        X_valid (numpy.ndarray): of shape(m, 784) containing the validation
        data
        Y_valid (numpy.ndarray): of shape(m, 10) containing the training labels
        batch_size (int): is the number of data points in a batch
        epochs (int): is the number of times the training should pass through
        the whole dataset
        load_path (str): is the path from which to load the model
        save_path (str): is the path to where the model should be saved after
        training

    Returns:
        the path where the model was saved
    """
    
