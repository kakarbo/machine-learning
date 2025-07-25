#!/usr/bin/env python3
"""
A layer with L2 Regularization
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization.

    Args:
        prev: Is a tensor containing the output of the previous layer.
        n: is the number of nodes the new layer should contain.
        activation: is the activation function that should be used on the layer.
        lambtha: is the L2 regularization parameter.
    
    Returns:
        A tensor that containing the new layer.
    """
    new_layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode=("fan_avg")
        ),
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )(prev)

    return new_layer
