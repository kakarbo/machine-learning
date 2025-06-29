#!/usr/bin/env python3
"""
L2 Regularization cost
"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
        cost (tensor): containing the cost the network withoud L2 regularization.
    
    Retunrs:
        a tensor containing the cost of the network accounting for L2 regularization
    """
    print(cost.numpy())
    return cost
