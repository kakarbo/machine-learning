#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig, ax = plt.subplots()

ax.hist(student_grades, bins=10, linewidth=0.5, edgecolor="black")

plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.savefig('Project_A')
