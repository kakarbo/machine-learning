#!/usr/bin/env python3
"""
En este ejercicio accedemos a las columnas 3 y 4 de la matriz
"""

matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
for vector in matrix:
    new_vector = []
    for element in range(len(vector)):
        if element == 2 or element == 3:
            new_vector.append(vector[element])
    the_middle.append(new_vector)
print("The middle columns of the matrix are: {}".format(the_middle))
