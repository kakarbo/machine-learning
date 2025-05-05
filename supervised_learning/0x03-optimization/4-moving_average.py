#!/usr/bin/env python3
"""
Moving Average
"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Parameters:
        data (list): is the list of data to calculate the moving average of
        beta (flota): is the weight used for the moving average
    
    Returns:
        A list containing the moving averages of data
    """
    # data = np.array(data, dtype=np.float64)
    # v = np.zeros_like(data, dtype=np.float64)
    # v[0] = data[0]

    v = 0
    EMA = []

    for t in range(len(data)):
        v = ((v * beta) + (1 - beta) * data[t])
        EMA.append(v / (1 - (beta ** (t + 1))))
    
    return EMA
