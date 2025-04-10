#!/usr/bin/env python3
"""
Train
"""
import tensorflow.compat.v1 as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def train(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    layer_zises,
    activations,
    alpha,
    iterations,
    save_path="/tmp/model.ckpt"
):
    """
    Builds, trains and saves a neural network classifier

    Parameters:
        X_train (numpy.ndarray): containing the training 
        input data
        Y_train (numpy.ndarray): containing the training
        labels
        X_valid (numpy.ndarray): containing the validation
        input data
        Y_valid (numpy.ndarray): containing the validation
        labels
        layer_zises (list): containing the number of nodes
        in each layer of the network
        activations (list): containing the activation
        functions for each layer of the network
        alpha (float): is the learning rate
        iterations (int): is the number of iterations
        to train over
        save_path (string): designates where to save
        the model

    Returns:
        the path where the model was saved
    """
    print(X_train[0], Y_train[0])
    print(X_valid[0], Y_valid[0])
