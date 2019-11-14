#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def load_data(fn):
  xs = []
  ys = []
  with open(fn, 'r', encoding='latin-1') as f:
    for ctr,line in enumerate(f):
      line = line.strip()
      data = list(line.split())
      if ctr == 0:
        fields = int(len(data)/4)
        for i in range(fields):
          xs.append([])
          ys.append([])
      xs[0].append(float(data[0]))
      ys[0].append(float(data[1]))

  return xs, ys

if __name__ == '__main__':
  fn = sys.argv[1]
  mode = sys.argv[2]
  data = load_data(fn)
  rc('font', **{'family': 'serif', 'serif':['Helvetica'], 'size':60})
  rc('text', usetex = True)
  rc('axes', lw = 5)
  rc('xtick.major', size = 20, width = 5)
  rc('xtick.minor', size = 16, width = 4)
  rc('ytick.major', size = 20, width = 5)
  rc('ytick.minor', size = 16, width = 4)

  fig, axes = plt.subplots(figsize =(30,30))
  
  print(data)
  if mode == 'fit':
    axes.loglog(data[0][1], data[1][1], 'bo', label = 'Small Particles')
    axes.loglog(data[0][0], data[1][0], label = 'fit', lw = 2, c = 'black')
    axes.set_xlabel(r'Q [$\AA^{-1}$]')
    axes.set_ylabel(r'Int [a.u.]')
  if mode == 'nr':
    axes.plot(data[0][0], np.multiply(np.array(data[1][0]), np.power(np.array(data[0][0]),3)), lw=10)
    axes.set_xlabel(r'Particle radius [nm]')
    axes.set_ylabel(r'N(r)*r$^3$ [nm$^3$]')
    
  ofn = fn.split('.')[0]+'.png'
  print(ofn)
  plt.savefig(ofn)
  #plt.show()