#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt

def load_data(fn):
  result = []
  with open(fn, 'r') as f:
    for line in f:
      data = line.split()
      result.append(data)
  
  return np.transpose(np.array(result,dtype=float))

if __name__ == '__main__':
  
  fn = sys.argv[1]
  
  data = load_data(fn)
  print(data)
  print(data[1])
  print(data[2])
  
  plt.style.use('ggplot')
  plt.rcParams.update({'font.size':22})
  figure = plt.figure(figsize=(12,8))
  
  plt.ylabel('Intensity [a.u.]')
  plt.xlabel(r'Q [nm$^{-1}$]')
  plt.plot(data[1],data[2])
  plt.xlim(11,49) 
  plt.ylim(400,1600)
  
  #plt.show()
  plt.savefig('{}.png'.format(fn[:-4]))
