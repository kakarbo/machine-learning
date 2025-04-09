#!/usr/bin/env python3
"""
Forward Propagation
"""
create_layer = __import__("1-create_layer").create_layer


def forward_prop(x, layer_size=[], activations=[]):
    """
    forward propagation

    Parameters
        x (placeholder): is the placeholder for the input data
        layer_size (int): is a list containing the number of
        nodes in each layer of the network
        activations (functions): is a list containing the
        activation functions for each layer of the network
    
    Returns
        Tensor: the prediction of the network in tensor form
    """
    for value in range(len(layer_size)):
        if value is 0:
            layer = create_layer(x, layer_size[value], activations[value])
        else:
            layer = create_layer(layer, layer_size[value], activations[value])
    
    return layer
