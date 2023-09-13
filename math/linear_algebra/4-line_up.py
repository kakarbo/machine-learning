#!/usr/bin/env python3
"""
Este modulo es para sumar dos matrices de igual tamaño por elementos
"""


def add_arrays(arr1, arr2):
    """
    Función que suma dos matrices por elementos
    """
    if len(arr1) != len(arr2):
        return None
    
    new_list = []
    for i in range(len(arr1)):
        new_list.append(arr1[i] + arr2[i])

    return new_list
