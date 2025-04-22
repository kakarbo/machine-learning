#!/usr/bin/env python3
"""
Train
"""
import tensorflow.compat.v1 as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def train(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    layer_zises,
    activations,
    alpha,
    iterations,
    save_path="/tmp/model.ckpt"
):
    """
    Builds, trains and saves a neural network classifier

    Parameters:
        X_train (numpy.ndarray): containing the training 
        input data
        Y_train (numpy.ndarray): containing the training
        labels
        X_valid (numpy.ndarray): containing the validation
        input data
        Y_valid (numpy.ndarray): containing the validation
        labels
        layer_zises (list): containing the number of nodes
        in each layer of the network
        activations (list): containing the activation
        functions for each layer of the network
        alpha (float): is the learning rate
        iterations (int): is the number of iterations
        to train over
        save_path (string): designates where to save
        the model

    Returns:
        the path where the model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_zises, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        for i in range(iterations):
            loss_train = session.run(
                loss,
                feed_dict={x: X_train, y: Y_train}
            )
            accuracy_train = session.run(
                accuracy,
                feed_dict={x: X_train, y: Y_train}
            )
            loss_valid = session.run(
                loss,
                feed_dict={x: X_valid, y: Y_valid}
            )
            accuracy_valid = session.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid}
            )
            if (i % 100) is 0:
                print(f"After {i} iterations")
                print(f"\tTraining Cost: {loss_train}")
                print(f"\tTraining Accuracy: {accuracy_train}")
                print(f"\tValidation Cost: {loss_valid}")
                print(f"\tValidation Accuracy: {accuracy_valid}")
            session.run(train_op, feed_dict={x: X_train, y: Y_train})

        i += 1
        loss_train = session.run(
            loss,
            feed_dict={x: X_train, y: Y_train}
        )
        accuracy_train = session.run(
            accuracy,
            feed_dict={x: X_train, y: Y_train}
        )
        loss_valid = session.run(
            loss,
            feed_dict={x: X_valid, y: Y_valid}
        )
        accuracy_valid = session.run(
            accuracy,
            feed_dict={x: X_valid, y: Y_valid}
        )
        print(f"After {i} iterations")
        print(f"\tTraining Cost: {loss_train}")
        print(f"\tTraining Accuracy: {accuracy_train}")
        print(f"\tValidation Cost: {loss_valid}")
        print(f"\tValidation Accuracy: {accuracy_valid}")

        return saver.save(session, save_path)
