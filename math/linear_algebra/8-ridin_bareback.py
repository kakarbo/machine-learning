#!/usr/bin/env python3
"""
Este modulo raliza la multiplicación de matrices
"""


def mat_mul(mat1, mat2):
    """
    Función que realiza la multiplicación de matrices
    """
    if len(mat1[0]) != len(mat2):
        return None

    final_list = []
    new_list = [element for sub_list in mat1 for element in sub_list]
    list_1 = mat2[0]
    list_2 = mat2[1]
    result_1 = []
    result_2 = []
    for num in new_list:
        if num % 2 == 0:
            for element in list_2:
                result_1.append(num * element)
        else:
            for element in list_1:
                result_2.append(num * element)
        if len(result_1) > 0 and len(result_2) > 0:
            result = [x + y for x, y in zip(result_1, result_2)]
            final_list.append(result)
            result_1 = []
            result_2 = []
    
    return final_list
