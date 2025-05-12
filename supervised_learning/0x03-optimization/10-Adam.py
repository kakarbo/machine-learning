#!/usr/bin/env python3
"""
Adam Upgraded
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using the
    Adam optimization algorithm

    Parameters:
        loss (float): is the loss of the network.
        alpha (float): is the learning rate.
        beta1 (float): is the weight used for the first moment
        beta2 (float): is the weight used for the second moment.
        epsilon (float): is a small number to avoid division by zero
    
    Returns:
        The Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
    train_op = optimizer.minimize(loss)

    return train_op
