#!/usr/bin/python3
#import hdf5plugin
#import h5py

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from os.path import join, isfile
from os import listdir

import argparse

def load_2d(fn):
  files = [join(fn,f) for f in listdir(fn) 
            if (isfile(join(fn,f))) and (f.endswith('.tif')) and not(f.endswith('corrected.tif'))]

  result = []
  
  for data in files:
    im = Image.open(data)
      result.append(np.array(im))
  
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot one or multiple h5 images.')  

  parser.add_argument('path', nargs=1 , type=str, help='Path to the files.',
                      metavar='PATH')
  parser.add_argument('inputfiles', nargs='+', type=str, help='Path to where the files are.',
                      metavar='INPUTFILES')
  parser.add_argument('-m', '--mode', type=str, choices=['mean','each'],
                      help='Plot mode. Either <mean> or <each>. Default is mean.', default ='mean')
  parser.add_argument('-l', '--log', type=bool, help='Plot with log scale.', default=False, 
                      const=True, nargs='?')

  args = parser.parse_args()
  print(args)
  path = args.path[0]
  files = args.inputfiles
  
  images = []
  
  for fn in files:
    if args.mode == 'mean':
      images.append(load_2d(join(path,fn)))
  
  if 
  
