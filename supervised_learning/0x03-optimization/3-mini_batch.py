#!/usr/bin/env python3
"""
Mini-Batch
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__("2-shuffle_data").shuffle_data


def train_mini_batch(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    batch_size=32,
    epochs=5,
    load_path="/tmp/model.ckpt",
    save_path="/tmp/model.ckpt"
):
    """
    Trains a loaded neural network model using mini-batch gradient descent

    Parameters:
        X_train (numpy.ndarray): of shape(m, 784) containing the training data
        Y_train (numpy.ndarray): of shape(m, 10) containing the training labels
        X_valid (numpy.ndarray): of shape(m, 784) containing the validation
        data
        Y_valid (numpy.ndarray): of shape(m, 10) containing the training labels
        batch_size (int): is the number of data points in a batch
        epochs (int): is the number of times the training should pass through
        the whole dataset
        load_path (str): is the path from which to load the model
        save_path (str): is the path to where the model should be saved after
        training

    Returns:
        the path where the model was saved
    """
    m = X_train.shape[0]
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("./graph.ckpt.meta")
        saver.restore(sess, './graph.ckpt')
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        
        for epoch in range(epochs + 1):
            print(f'After {epoch} epochs:')

            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            print(f'\tTraining Cost: {train_cost}')

            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            print(f'\tTraining Accuracy: {train_accuracy}')

            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            print(f'\tValidation Cost: {valid_cost} ')

            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            print(f'\tValidation accuracy: {valid_accuracy}')

            if epoch == epochs:
                break
            
            X_train_s, Y_train_s = shuffle_data(X_train, Y_train)
            if (m % batch_size) is 0:
                mini_batch_total = m // batch_size
            else:
                mini_batch_total = (m // batch_size) + 1
            
            step_number = 0
            for mini_batch in range(mini_batch_total):
                low = mini_batch + batch_size
                high = ((mini_batch + 1) * batch_size)
                if high > m:
                    high = m
                sess.run(train_op, feed_dict={x: X_train_s[low:high, :], y: Y_train_s[low:high, :]})

                step_number += 1
                if (step_number % 100) is 0:
                    print(f'\tStep {step_number}:')
                    step_cost = sess.run(loss, feed_dict={x: X_train_s[low:high, :], y: Y_train_s[low:high, :]})
                    print(f'\t\tCost: {step_cost}')

                    step_accuracy = sess.run(accuracy, feed_dict={x: X_train_s[low:high, :], y: Y_train_s[low:high, :]})
                    print(f'\t\tAccuracy: {step_accuracy}')

        return saver.save(sess, save_path)