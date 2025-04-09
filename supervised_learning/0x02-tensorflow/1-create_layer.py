#!/usr/bin/env python3
"""
This module is responsible for creating layers for neural networks.
It provides functionality to define, initialize, and manage
different types of layers, ensuring proper structure and
parameterization for deep learning models. It supports various
activation functions, weight initialization methods, and layer
configurations to facilitate efficient model training and inference.
"""
import tensorflow.compat.v1 as tf

def create_layer(prev, n, activation):
    """
    This function dynamically generates neural network layers
    with configurable depth and activation, commonly used in sequential models

    Parameters:
        prev (tensor): is tensor output of the previous layer.
        n (int): is the number of nodes in the layer to create.
        activation (fuction): is the activation function that the layer should use
    
    Returns:
        tensor: the tensor ouput of the layer
    """
    initializer = tf.initializers.variance_scaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=initializer, name="layer")
    return layer(prev)
