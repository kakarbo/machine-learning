#!/usr/bin/env python3
"""
Train op
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network

    Parameters:
        loss (tensor): is the loss of the network's prediction
        alpha (float): is the learning rate

    Returns:
        an operation that trains the network using gradient
        descent
    """
    op = tf.train.GradientDescentOptimizer(
        learning_rate=alpha,
        name="GradientDescent"
    )
    print(loss)
    return op.apply_gradients(op.compute_gradients(loss))
