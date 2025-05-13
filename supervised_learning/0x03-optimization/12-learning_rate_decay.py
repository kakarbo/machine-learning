#!/sur/bin/env python3
"""
Learning Rate Decay Upgraded
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation in tensorflow using inverse time
    decay.

    Parameters:
        alpha (float): is the original learning rate
        decay_rate (int): is the weight used to determine the rate at which
        alpha will decay
        global_step (int): is the number of passes of gradient descent that
        have elapses
        decay_step (int): is the number of passes of gradient descent that
        should occur before alpha is decayed further

    Returns:
        (float) - The learning rate decay operation
    """
    step = tf.floor(tf.cast(global_step, tf.float32) / decay_step)
    decayed_alpha = alpha / (1 + decay_rate * step)

    return decayed_alpha
