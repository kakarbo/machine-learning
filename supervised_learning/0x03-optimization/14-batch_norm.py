#!/usr/bin/env python3
"""
Batch Normalization Upgraded
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Parameters:
        prev: is the activated output of the previos layer.
        n (int): is the number of nodes in the layer to be created.
        activation: is the activation function that should be used on the
        output of the layer
    you should use the tf.keras.layers.Dense layer as the base layer with kernal initializer tf.keras.initializers.VarianceScaling(mode='fan_avg')
your layer should incorporate two trainable parameters, gamma and beta, initialized as vectors of 1 and 0 respectively
you should use an epsilon of 1e-8
    Returns:
        A tensor of the activated output for the layer
    """
    kernel_initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.layers.dense(
        inputs=prev,
        units=n,
        kernel_initializer=kernel_initializer,
        use_bias=False
    )

    batch_mean, batch_var = tf.nn.moments(dense_layer, axes=[0], keep_dims=True)
    gamma = tf.Variable(tf.ones([n]), name='gamma', trainable=True)
    beta = tf.Variable(tf.zeros([n]), name='beta', trainable=True)

    epsilon = 1e-8
    z_norm = tf.divide(
        dense_layer - batch_mean,
        tf.sqrt(batch_var + epsilon)
    )

    batch_norm_output = gamma * z_norm

    return activation(batch_norm_output)
