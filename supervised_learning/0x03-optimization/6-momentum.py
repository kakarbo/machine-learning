#!/usr/bin/env python3
"""
Momentum upgraded
"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for neural network in tensorflow using the
    gradiente descent with momentum optimization algorithm

    Parameters:
        loss (tensorflow): is the loss of the network
        alpha (float): is the learning rate
        beta1 (float): is the momentum weight

    Returns:
        The momentum optimization operation
    """
    op = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

    return op
