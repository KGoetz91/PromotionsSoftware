#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt

def load_uvVis(fn):
  
  result = {}
  
  with open(fn, 'r') as f:
    for ctr, line in enumerate(f):
      if ctr > 6:
        x, time, i = line.split()
        x = x.replace(',', '.')
        i = i.replace(',', '.')
        
        if time in result.keys():
          result[time]['x'].append(float(x))
          result[time]['y'].append(float(i))
        else:
          result[time] = {'x': [float(x)], 'y': [float(i)]}
  y = np.array([])
  for key in result.keys():
    x = result[key]['x']
    if len(y) == 0:
      y = np.array(result[key]['y'])
    else:
      y = np.add(y, np.array(result[key]['y']))
    
  y = np.divide(y, len(result))  
  return (x, y)


if __name__ == '__main__':

  fn_dt = sys.argv[1]
  fn_bg = sys.argv[2]
  factor = float(sys.argv[3])
  data = load_uvVis(fn_dt)
  bg = load_uvVis(fn_bg)
  
  plt.plot(data[0], data[1])
  plt.plot(bg[0], np.multiply(bg[1], factor))
  plt.show()
