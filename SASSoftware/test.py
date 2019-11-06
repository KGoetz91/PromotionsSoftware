#!/usr/bin/python3

from formfactors import Cube, CubeOneShell
import matplotlib.pyplot as plt
import numpy as np

cube = CubeOneShell(1e-7,1, 0.5,0,4,1)
#cube = Cube(1e-7,1, 0,5)

steps = 1000
max_q = 20
min_q = 0

q_range = range(1000)
q_range = np.multiply(np.add(np.divide(np.array(q_range), 1000), min_q), max_q-min_q)

i = cube.scatter(q_range)
print('debug')
i = np.array(i)
i = np.transpose(i)

plt.loglog(i[0], i[1])
plt.show()
