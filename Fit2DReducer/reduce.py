#!/usr/bin/python3

import argparser
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def loadChi(fn):
  with open(fn, 'r') as f:
    xs = []
    ys = []
    for ctr, line in enumerate(f):
      if ctr > 3:
        line = line.strip()
        x, y = line.split()
        xs.append(float(x))
        ys.append(float(y))
        
  return (xs,ys)

def loadTIFF(fn):
  return np.array(Image.open(fn))

def calc_CF(std, gc)

if __name__ == '__main__':

  #parser = argparse.ArgumentParser(description='Reduce .chi files from fit2d')
  
  #parser.add_argument('-gc', '--glassy-carbon-files', dest='gcfiles', type=str, nargs='+', help='Glassy carbon files to use. Needs standard file, measured file, transmission and direct beam or path to a file where the paths to those files are stored in seperate lines.')
  #parser.add_argument('-d', '--data-files', dest='dfiles', type=str, nargs='+', help='Data files to use. Needs measured file, transmission and direct beam or path to a file where the paths to those files are stored in seperate lines.')
  
  mode = sys.argv[1]
  
  if mode == 'gc':
    std_gc = sys.argv[2]
    gc = sys.argv[3]
    gc_tm = sys.argv[4]
    gc_db = sys.argv[5]
    time = sys.argv[6]
  
  data = 
