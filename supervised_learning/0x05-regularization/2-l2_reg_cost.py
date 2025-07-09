#!/usr/bin/env python3
"""
L2 Regularization cost
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: A tensor containing the cost of the network without L2
            regularization.
    
    Returns:
        A tensor containing the cost of the network accounting for L2
        regularization.
    """
    return cost + tf.losses.get_regularization_loss()
