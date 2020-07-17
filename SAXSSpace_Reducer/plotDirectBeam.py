#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

from PIL import Image

from os.path import join, isfile
from os import listdir

import sys

def load_2d(fn):
  files = [join(fn,f) for f in listdir(fn) 
            if (isfile(join(fn,f))) and (f.endswith('.tif')) and not(f.endswith('corrected.tif'))]

  result = []
  
  for data in files:
    im = Image.open(data)
    result.append(np.array(im))
  
  return result

def two_d_gaussian(x1, x2, mu1, mu2, sigma1, sigma2, rho, amplitude):
  drho = 1-rho**2
  factor = 1/(2*np.pi*sigma1*sigma2*np.sqrt(drho))
  d1 = x1-mu1
  d2 = x2-mu2
  exponential = np.exp(-(1/(2*drho))*((d1**2/sigma1**2)+(d2**2/sigma2**2)-((2*rho*d1*d2)/(sigma1*sigma2))))
  
  return amplitude*factor*exponential

if __name__ == '__main__':
  
  path = sys.argv[1]
  images = load_2d(path)
  
  com = ndimage.measurements.center_of_mass(images[0])
  xcom=int(com[0])-6
  ycom=int(com[1])
  
  xprojection = np.sum(images[0],0)
  yprojection = np.sum(images[0],1)
  plt.plot(np.log10(xprojection))
  plt.plot(np.log10(yprojection))
  plt.show()
  
  width=20
  
  coms = []
  
  for i in range(len(images)):
    coms.append(ndimage.measurements.center_of_mass(images[i]))
    images[i] = images[i][xcom-width:xcom+width,ycom-width:ycom+width]
  
  size = int(np.sqrt(len(images)))+1
  
  fig, ax = plt.subplots(size, size)
  ax = ax.flatten()
  
  for ctr,i in enumerate(images):
    ax[ctr].imshow(np.log(i))
    ax[ctr].annotate('({:.2f}:{:.2f})'.format(coms[ctr][0],coms[ctr][1]),
            xy=(coms[ctr][1]-ycom+20, coms[ctr][0]-xcom+20), xycoords='data',
            xytext=(-20, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
    #ax[ctr].text(coms[ctr][0]-xcom+20, coms[ctr][1]-ycom+20, 'x')
  plt.show()