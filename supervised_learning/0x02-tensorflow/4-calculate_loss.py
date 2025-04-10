#!/usr/bin/env python3
"""
Loss
"""
import tensorflow.compat.v1 as tf

def calculate_loss(y, y_pred):
    """
    Calculate the softmax cross-entropy loss of a prediction

    Parameters:
        y (placeholder): is a placeholder for the labels
        of the input data
        y_pred (tensor): is a tensor containing the 
        network's predictions

    Returns:
        Tensor: a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)