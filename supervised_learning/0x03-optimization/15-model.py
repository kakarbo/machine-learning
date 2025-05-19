#!/usr/bin/env python3
"""
Put it all together and what do you get
"""
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense, BatchNormalization


def forward_prop(prev, layers, activations, epsilon):
    #all layers get batch_normalization but the last one, that stays without any activation or normalization
    x = prev

    for i in range(len(layers)):
        x = Dense(layers[i])(x)

        if i != len(layers) - 1:
            x = BatchNormalization(epsilon=epsilon)(x)
            if activations[i] is not None:
                x = activations[i](x)
    
    return x


def shuffle_data(X, Y):
    # fill the function
    m, nx = X.shape
    indices = np.random.permutation(m)

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    return X_shuffled, Y_shuffled


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow using Adam optimization
    mini-batch gradient descent, learning rate decay, and batch normalization.

    Parameters:
        Data_train (tuple): containing the training inputs and training labels,
        respectively
        Data_valid (tuple): containing the validation inputs and validation labels,
        respectively
        layers (list): containing the number of nodes in each layer of the network
        activation (list): containing the activation functions used for each layer of 
        the network
        alpha (float): is the learning rate
        beta1 (float): is the weight for the first moment of Adam optimization
        beta1 (float): is the wight for the second moment of Adam optimization
        epsilon (int): is a small number used to avoid division by zero
        decay_rate (int): is the decay rate for inverse time decay of the learning
        rate
        batch_size (int): is the number of data points that should be in a mini-batch
        epochs (int): is the number of times the training should pass through the whole
        dataset
        save_path (str): is the path where the model should be saved to
    
    Returns:
        str - the path where the model was saved
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float64, shape=(None, 784), name="x")
    y = tf.placeholder(tf.float64, shape=(None, 10), name="y")

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)

    # intialize loss and add it to collection
    loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            y, y_pred, from_logits=True
        )
    )
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(y, axis=1),
                tf.argmax(y_pred, axis=1)
            ),
            tf.float64
        )
    )
    tf.add_to_collection('accurray', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)
    
    # compute decay_steps
    n, _ = X_train.shape
    decay_steps = (n // batch_size) * epochs
    
    # create "alpha" the learning rate decay operation in tensorflow
    return 'path'

    # initizalize train_op and add it to collection 
    # hint: don't forget to add global_step parameter in optimizer().minimize()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy

            # shuffle data
            pass

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                pass

                # run training operation

                                # print batch cost and accuracy

        # print training and validation cost and accuracy again

        # save and return the path to where the model was saved
