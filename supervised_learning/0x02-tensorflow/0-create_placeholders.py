#!/usr/bin/env python3
"""
In TensorFlow 1.x, placeholders were a mechanism to define
inputs to a computational graph. They acted as "promises"
to provide data later, during the execution of the graph
via a Session. However, TensorFlow 2.x deprecated
placeholders in favor of a more intuitive approach
(eager execution and tf.function). Here's a detailed
breakdown
"""
import tensorflow.compat.v1 as tf

def create_placeholders(nx, classes):
    """
    create placeholders with tensorflow

    Parameters:
        nx (float): the number of feature columns in our data
        classes (float): the number of classes in our classifier

    Returns:
        (x, y) (tupla(Tensor, Tensor)): returns two placeholders x and y,
        for the neural network
    """

    # disabling eager mode
    tf.compat.v1.disable_eager_execution()

    # Creating a tensorflow graph
    graph = tf.Graph()

    with graph.as_default():

        # creating a placeholder
        x = tf.placeholder(tf.float64, (0, nx), name="x")
        y = tf.placeholder(tf.float64, (0, classes), name="y")

        return (x, y)


