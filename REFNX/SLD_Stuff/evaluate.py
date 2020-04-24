#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def load_refnx_sld(fn):
  xs=[]
  ys=[]
  
  with open(fn) as f:
    for line in f:
      x,y = line.strip().split()
      xs.append(float(x))
      ys.append(float(y))
      
  return xs,ys      

if __name__ == '__main__':
  path = sys.argv[1]
  fullSLD = sys.argv[2]
  onlysub = sys.argv[3]
  onlylayer = sys.argv[4]
  
  outn = sys.argv[5]

  fSLD = list(load_refnx_sld(os.path.join(path,fullSLD)))
  subSLD = list(load_refnx_sld(os.path.join(path,onlysub)))
  layerSLD = list(load_refnx_sld(os.path.join(path,onlylayer)))
  
  plt.plot(fSLD[0], fSLD[1])
  plt.plot(subSLD[0], subSLD[1])
  plt.plot(subSLD[0], np.array(fSLD[1])-np.array(subSLD[1]))
  plt.plot(layerSLD[0], layerSLD[1])
  
  plt.show()
  
  with open(os.path.join(path,outn), 'w') as of:
    outdata = zip(subSLD[0], np.array(fSLD[1])-np.array(subSLD[1]))
    for x,y in outdata:
      of.write(str(x)+' '+str(y)+'\n')
