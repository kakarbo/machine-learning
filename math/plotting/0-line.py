#!/usr/bin/env python3
'''
Este modulo genera un grafico lineal
'''

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)

# crear un grafico de lineas
ax.plot(x, y0, color='red', linestyle='-')
plt.savefig('line-graph.jpg')
