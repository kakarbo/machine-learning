#!/usr/bin/env python3

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
l2_reg_cost = __import__("2-l2_reg_cost").l2_reg_cost


def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    m = Y.shape[0]
    oh = np.zeros((classes, m))
    oh[Y, np.arange(m)] = 1
    return oh


np.random.seed(4)
m = np.random.randint(1000, 2000)
c = 10
lib = np.load("../data/MNIST.npz")

X = lib["X_train"][:m].reshape((m, -1))
Y = one_hot(lib["Y_train"][:m], c).T

n0 = X.shape[1]
n1, n2 = np.random.randint(10, 1000, 2)

lam = 0.09
tf.set_random_seed(0)

x = tf.placeholder(tf.float32, (None, n0))
y = tf.placeholder(tf.float32, (None, c))

a1 = tf.keras.layers.Dense(
    n1,
    activation=tf.nn.tanh,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg")
    ),
    kernel_regularizer=tf.keras.regularizers.L2(lam),
)(x)
a2 = tf.keras.layers.Dense(
    n2,
    activation=tf.nn.sigmoid,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg")
    ),
    kernel_regularizer=tf.keras.regularizers.L2(lam),
)(a1)
y_pred = tf.keras.layers.Dense(
    c,
    activation=None,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg")
    ),
    kernel_regularizer=tf.keras.regularizers.L2(lam),
)(a2)

cost = tf.losses.softmax_cross_entropy(y, y_pred)

l2_cost = l2_reg_cost(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(l2_cost, feed_dict={x: X, y: Y}))