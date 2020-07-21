#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 
from scipy.optimize import curve_fit as cfit

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

def two_d_gaussian( x, mu1, mu2, sigma1, sigma2, rho, amplitude):
  drho = 1-rho**2
  factor = 1/(2*np.pi*sigma1*sigma2*np.sqrt(drho))
  
  dx = np.subtract(x,(mu1, mu2))
  dexp = np.array([ (d1**2/sigma1**2 + d2**2/sigma2**2 - (2*rho*d1*d2)/(sigma1*sigma2)) for d1, d2 in dx])
  fexp = -1/(2*drho)
  exp_int = np.multiply(fexp, dexp)
  exponential = np.exp(exp_int)
  
  result = np.multiply(amplitude*factor, exponential)
  
  return result

if __name__ == '__main__':
  
  path = sys.argv[1]
  images = load_2d(path)
  
  com = ndimage.measurements.center_of_mass(images[0])
  xcom=int(com[0])-6
  ycom=int(com[1])
  print(xcom)
  print(ycom)
  

  width=20
  xs = [(i,j) for j in range(2*width) for i in range(2*width)]
  #xs = [(i,j) for i in range(2*width) for j in range(2*width)]
  #print(xs)
  
  coms = []
  
  for i in range(len(images)):
    #coms.append(ndimage.measurements.center_of_mass(images[i]))
    images[i] = images[i][xcom-width:xcom+width,ycom-width:ycom+width]
    
    intensities = images[i].flatten()

    p0 = [ 20, 20, 0.2,0.1,0.2,8e4]
    l_b = [18, 18, 0.01, 0.01, -0.99, 0]
    u_b = [22, 22, 1, 1, 0.99, np.inf]

    popt, pcov = cfit(two_d_gaussian, xs, intensities, p0, bounds=(l_b,u_b))
    coms.append((popt[1]+ycom,popt[0]+xcom))
    print(popt)

    #ti = two_d_gaussian(xs, *popt)

    #ti = np.reshape(ti, (40,40))
    #fig, ax = plt.subplots(2,2)
    #ax = ax.flatten()
    
    #ax[0].imshow(np.log10(np.reshape(intensities,(40,40))))
    #ax[1].imshow(np.log10(ti))
    
    #ax[2].imshow(np.reshape(intensities,(40,40)))
    #ax[3].imshow(ti)
    #plt.show()

  size = int(np.sqrt(len(images)))+1
  
  fig, ax = plt.subplots(size, size)
  ax = ax.flatten()
  
  print(coms)
  
  for ctr,i in enumerate(images):
    ax[ctr].imshow(np.log(i))
    ax[ctr].annotate('({:.2f}:{:.2f})'.format(coms[ctr][1],coms[ctr][0]),
            xy=(coms[ctr][1]-xcom, coms[ctr][0]-ycom), xycoords='data',
            xytext=(-20, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"))
    #ax[ctr].text(coms[ctr][0]-xcom+20, coms[ctr][1]-ycom+20, 'x')
  plt.show()
