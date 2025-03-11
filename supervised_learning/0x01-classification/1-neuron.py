#!/usr/bin/env python3
Neuron =  __import__('0-neuron').Neuron
import numpy as np
"""
Binary Classification
"""

class Neuron(Neuron):
    """
    Defined a single neuron performing binary classification
    """
    __W = Neuron.W
    __b = 0
    __A = 0
    def __init__(self, nx):
        """
        class constructor
        """
        super().__init__(nx)

