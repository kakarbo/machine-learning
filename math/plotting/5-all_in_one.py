#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x0 = np.arange(0, 11)
y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

fig = plt.figure(figsize=(12, 8))

small_font_size = 'x-small'

ax1 = plt.subplot2grid((3, 2), (0, 0))
ax1.plot(x0, y0, color='red', linestyle='-')

ax2 = plt.subplot2grid((3, 2), (0, 1))
ax2.scatter(x1, y1, c='magenta')
ax2.set_xlabel('Height (in)', fontsize=small_font_size)
ax2.set_ylabel('Weight (lbs)', fontsize=small_font_size)
ax2.set_title('Men\'s Height vs Weight', fontsize=small_font_size)

ax3 = plt.subplot2grid((3, 2), (1, 0))
ax3.semilogy(x2, y2, c='blue')
ax3.set_xlabel('Time (years)', fontsize=small_font_size)
ax3.set_ylabel('Fraction Remaining', fontsize=small_font_size)
ax3.set_title('Exponential Decay of C-14', fontsize=small_font_size)

ax4 = plt.subplot2grid((3, 2), (1, 1))
ax4.plot(x3, y31, label='C-14', c='red', linestyle='--')
ax4.plot(x3, y32, label='Ra-226', c='green', linestyle='-')
ax4.set_xlabel('Time (years)', fontsize=small_font_size)
ax4.set_ylabel('Fraction Remaining', fontsize=small_font_size)
ax4.set_title('Exponential Decay of Radioactive Elements', fontsize=small_font_size)

ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
ax5.hist(student_grades, bins=10, linewidth=0.5, edgecolor="black")
ax5.set_xlabel('Grades', fontsize=small_font_size)
ax5.set_ylabel('Number of Students', fontsize=small_font_size)
ax5.set_title('Project A', fontsize=small_font_size)

fig.tight_layout()

plt.suptitle('All in One')

plt.savefig('All_in_One.jpg')
