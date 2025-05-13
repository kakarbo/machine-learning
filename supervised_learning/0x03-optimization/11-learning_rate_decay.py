#!/usr/bin/env python3
"""
Learning rate decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Parameters:
        alpha (float): is the original learning rate
        decay_rate (int): is the weight used to determine the rate at which alpha
        will decay
        global_step (int): is the number of passes of gradient descent that have
        elapsed.
        decay_step (int): is the number of passes of gradient descent that should
        occur before alpha is decayed further.

    Returns:
        the updated value for alpha
    """
    decayed_alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))

    return decayed_alpha
