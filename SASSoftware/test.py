#!/usr/bin/python3

from formfactors import Cube, Intensities
import matplotlib.pyplot as plt
import numpy as np

cube = Intensities(Cube(1e-7,1,0,5))
q_range = range(20)

i = cube.intensity(q_range)

i = np.array(i)
i = np.transpose(i)

plt.loglog(i[0], i[1])
plt.show()
