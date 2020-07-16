#!/usr/bin/python3

from PIL import Image

import pyFAI

from os import listdir
from os.path import join, isfile

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

import argparse

__CALIB__ = 'temp.poni'
__DIST__ = 0.3075
__CF__ = 1

def write_poni(distance, COM):
  with open(__CALIB__, 'w') as of:
    of.write('#pyFAI Calibration file constructed manually\n')
    of.write('#Created never...\n')
    of.write('poni_version: 2\n')
    of.write('Detector: Detector\n')
    of.write('Detector_config: {"pixel1": 7.5e-5, "pixel2": 7.5e-5, "max_shape": null}\n')
    of.write('Distance: {}\n'.format(distance))
    of.write('Poni1: {}\n'.format(COM[0]*7.5e-5))
    of.write('Poni2: {}\n'.format(COM[1]*7.5e-5))
    of.write('Rot1: 0\n')
    of.write('Rot2: 0\n')
    of.write('Rot3: 0\n')
    of.write('Wavelength: 1.5406e-10\n')

def load_data(fn):
  files = [join(fn,f) for f in listdir(fn) 
            if (isfile(join(fn,f))) and (f.endswith('.tif')) and not(f.endswith('corrected.tif'))]

  result = []
  
  for data in files:
    im = Image.open(data)
    if not len(result)>0:
      result = np.array(im)
    else:
      result = np.add(result, np.array(im))

  #plt.imshow((result))
  #plt.imshow(np.log(result))
  #plt.show()
  return result
  
def create_mask(data,com):
  mask = np.array(data)
  mask[mask >= 0] = 0
  mask[mask < 0] = 1
  mask[666,388] = 1
  mask[764,409] = 1
  mask[1006,274] = 1
  mask[int(com[0]):]=1
  np.save('mask.npy', mask)
  
def radial_integration(distance, com, data, outp):
  mask = np.load('mask.npy')
  write_poni(distance, com)
  ai = pyFAI.load(__CALIB__)
  res = ai.integrate1d(data, 1000, unit='q_nm^-1', mask = mask, filename='{}_1000bins.dat'.format(outp), error_model = 'poisson')
  res = ai.integrate1d(data, 100, unit='q_nm^-1', mask = mask, filename='{}_100bins.dat'.format(outp), error_model = 'poisson')
  res = ai.integrate1d(data, 200, unit='q_nm^-1', mask = mask, filename='{}_200bins.dat'.format(outp), error_model = 'poisson')

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Plot one or multiple h5 images.')  

  parser.add_argument('path', nargs=1 , type=str, help='Path to the files.',
                      metavar='PATH')
  parser.add_argument('inputfiles', nargs='+', type=str, help='Path to where the files are.',
                      metavar='INPUTFILES')
  parser.add_argument('-cf', type=float, help='Calibration factor. The data will be multiplied by this.',
                      default=__CF__)
  parser.add_argument('-d', type=float, help='Sample-Detector distance. Default is 0.3075 m.',
                      default=__DIST__)
  parser.add_argument('-o', type=str, help='Name of the output file. Default is integrated.',
                      default='integrated')
  parser.add_argument('-t', type=float, help='Measurement time. Default is 1 second.',
                      default=1)  
  parser.add_argument('-s', type=float, help='Sample thickness. Default is 1 mm.',
                      default=1)
  parser.add_argument('-i0', type=float, 
                      help='Flux of the incoming beam in counts per second. Default is 1.', default=1)

  args = parser.parse_args()
  print(args)

  path = args.path[0]
  files = args.inputfiles
  cf = args.cf
  time = args.t
  thick = args.s
  dist = args.d 
  I0 = args.i0
 
  for fn in files:
    outp = join(path,fn,args.o) 
    data = load_data(join(path,fn))
    trans = np.sum(data)/(time*I0)
    com = ndimage.measurements.center_of_mass(data)
    #plt.imshow(np.log(data))
    #plt.show()
    create_mask(data,com)
    data = np.multiply(data,(cf/(trans*time*thick)))
    radial_integration(dist, com, data, outp)
