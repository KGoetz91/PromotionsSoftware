#!/usr/bin/python3

from formfactors import Cube, Intensities
import matplotlib.pyplot as plt

cube = Intensities(Cube(1e-7,1,0,5))
q_range = range(100)

i = cube.intensity(q_range)

print(i)

plt.loglog(q_range, i)
plt.show()
