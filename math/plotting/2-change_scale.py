#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

ax.semilogy(x, y, c='blue')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.savefig('change_scale.jpg')
