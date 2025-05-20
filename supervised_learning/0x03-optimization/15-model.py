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
    tf.add_to_collection("y_pred", y_pred)

    # intialize loss and add it to collection
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred
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
    alpha_decayed = tf.train.exponential_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
        name="alpha_decayed"
    )

    # initizalize train_op and add it to collection 
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    optimizer = tf.train.AdamOptimizer(
        alpha,
        beta1,
        beta2,
        epsilon
    )
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy
            loss_train = sess.run(
                loss, feed_dict={x: X_train, y: Y_train}
            )
            accuracy_train = sess.run(
                accuracy, feed_dict={x: X_train, y: Y_train}
            )
            loss_valid = sess.run(
                loss, feed_dict={x: X_train, y: Y_train}
            )
            accuracy_valid = sess.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid}
            )
            print(f"After {i} iterations")
            print(f"\tTraining Cost: {loss_train}")
            print(f"\tTraining Accuracy: {accuracy_train}")
            print(f"\tValidation Cost: {loss_valid}")
            print(f"\tValidation Accuracy: {accuracy_valid}")

            # shuffle data
            X_train_s, Y_train_s = shuffle_data(X_train, Y_train)
            step_number = 0
            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and Y_train shuffled
                X_batch = X_train_s[j:j + batch_size]
                Y_batch = Y_train_s[j:j + batch_size]

                # run training operation
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})
                step_number += 1
                if (step_number % 100) is 0:

                            # print batch cost and accuracy
                            print(f"\tStep {step_number}:")
                            step_cost = sess.run(
                                loss, feed_dict={x: X_batch, y: Y_batch}
                            )
                            print(f"\t\tCost: {step_cost}")
                            step_accuracy = sess.run(
                                accuracy, feed_dict={x: X_batch, y: Y_batch}
                            )
                            print(f"\t\tAccuracy {step_accuracy}")


        # print training and validation cost and accuracy again
        sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        i += 1
        loss_train = sess.run(
            loss, feed_dict={x: X_train, y: Y_train}
        )
        accuracy_train = sess.run(
            accuracy, feed_dict={x: X_train, y: Y_train}
        )
        loss_valid = sess.run(
            loss, feed_dict={x: X_train, y: Y_train}
        )
        accuracy_valid = sess.run(
            accuracy, feed_dict={x: X_valid, y: Y_valid}
        )
        print(f"After {i} iterations")
        print(f"\tTraining Cost: {loss_train}")
        print(f"\tTraining Accuracy: {accuracy_train}")
        print(f"\tValidation Cost: {loss_valid}")
        print(f"\tValidation Accuracy: {accuracy_valid}")

        # save and return the path to where the model was saved
        return saver.save(sess, save_path)
