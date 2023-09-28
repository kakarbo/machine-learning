#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)


fig, ax = plt.subplots()
ax.plot(x, y1, label='C-14', c='red', linestyle='--')
ax.plot(x, y2, label='Ra-226', c='green', linestyle='-')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.savefig('two_is_better_than_one.jpg')
