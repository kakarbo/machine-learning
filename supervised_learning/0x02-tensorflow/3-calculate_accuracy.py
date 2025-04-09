#!/usr/bin/env python3
"""
Accuracy
"""
import tensorflow.compat.v1 as tf

def calculate_accuracy(y, y_pred):
    """
    Calculate accuracy

    Parameters:
        y (placeholder): is a placeholder for the labels of
        the input data
        y_pred (tensor): is a tensor containing the network's
        predictions

    Returns:
        Tensor: a tensor containing the decimal accuracy of
        the prediction
    """
    pred = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    return accuracy
