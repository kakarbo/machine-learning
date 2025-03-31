#!/usr/bin/env python3
"""

"""
import tensorflow.compat.v1 as tf

def create_placeholders(nx, classes):
    """
    create placeholders with tensorflow

    Parameters:
        nx (float): the number of feature columns in our data
        classes (float): the number of classes in our classifier

    Returns:
        (x, y) (tupla): returns two placeholders x and y,
        for the neural network
    """

    # disabling eager mode
    tf.compat.v1.disable_eager_execution()

    # Creating a tensorflow graph
    graph = tf.Graph()

    with graph.as_default():

        # creating a placeholder
        x = tf.placeholder(tf.float64, (, nx), name="x")
        y = tf.placeholder(tf.float64, (, classes), name="y")

        return (x, y)


