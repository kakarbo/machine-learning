#!/usr/bin/env python3
"""
Layer with Dropout
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout:

    Parameters:
        prev: is a tensor containing the output of the previous layer
        n: is the number of nodes the new layer should contain
        activation: is the activation function that should be used on the layer
        keep_prob: is the probability that a node will be kept
    
    Returns:
        the output of the new layer
    """
    new_layer = tf.keras.layers.Dense(
        n,
        activation
    )
    drop = tf.keras.layers.Dropout(rate=(1 - keep_prob))
    return drop(new_layer(prev))
    
