#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def load_data(fn):
  xs = []
  ys = []
  with open(fn, 'r', encoding='latin-1') as f:
    header = 50000
    for ctr,line in enumerate(f):
      line = line.strip()
      if line == '[Scan points]':
        header = ctr
      if ctr > header+1:
        line = line.replace(',', '.')
        if len(line) > 0:
          x,y = line.split(';')
          xs.append(float(x))
          ys.append(float(y))

  return xs, ys

if __name__ == '__main__':
  fn = sys.argv[1]
  data = load_data(fn)
  rc('font', **{'family': 'serif', 'serif':['Helvetica']})
  rc('text', usetex = True)
  fig, axes = plt.subplots()
  axes.plot(data[0], data[1])
  axes.set_xlabel(r'2$\theta$ [°]')
  ofn = fn.split('.')[0]+'.png'
  print(ofn)
  plt.savefig(ofn)
  #plt.show()
