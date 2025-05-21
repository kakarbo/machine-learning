#!/usr/bin/env python3
"""
Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
        labels (numpy.ndarray): is a one-hot of shape(m, classes) containing
        the correct labels for each data point.
            m (int): is the number of data points.
            classes (int): is the number of classes.
        logits (numpy.ndarray): is a one-hot of shape(m, classes) containing
        the predicted labels

    Returns:
        numpy.ndarray - a confusion of shape(classes, classes) with row
        indices representing the correct labels and column indices
        representing the predicted lables
    """
    # Convertir one-hot a indices de clases (posición del 1 en cada fila)
    true_labels = np.argmax(labels, axis=1)
    predicted_lables = np.argmax(logits, axis=1)

    # Numero de clases
    num_classes = labels.shape[1]

    # Inicializar matriz de confusión (true x predicted)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Llenar la matriz contando coincidencias (true_label, predicted_label)
    np.add.at(confusion_matrix, (true_labels, predicted_lables), 1)

    return confusion_matrix
