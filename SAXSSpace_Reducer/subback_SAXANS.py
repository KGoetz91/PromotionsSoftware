#!/usr/bin/python3

from PIL import Image

import pyFAI

from os import listdir
from os.path import join, isfile

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

import argparse

def load_data(fn):
  xs = []
  ys = []
  es = []

  with open(fn, 'r') as f:
    for line in f:
      try:
        if not line.startswith('#'):
          data = list(line.split())
          xs.append(float(data[0]))
          ys.append(float(data[1]))
          es.append(float(data[2]))
      except:
        #print('Skipping line: {}'.format(line))
        pass

  result = np.array([xs,ys,es])
  print(result)
  return result

def lin_interpolate(x1,y1,e1,x2,y2,e2,x_det):
  a = (y2-y1)/(x2-x1)
  ea = np.sqrt((e1*e1)+(e2*e2))
  eb = np.sqrt((e1*e1)+(e2*e2)+(ea*ea))
  b = 0.5*(y2+y1-a*(x1+x2))
  ydet = a*x_det+b
  edet = np.sqrt((x_det*x_det*ea*ea)+(eb*eb))
  return x_det,ydet,edet

def spline_data(x0, data):
  x1 = np.array(data[0])
  y1 = np.array(data[1])
  e1 = np.array(data[2])

  y2 = []
  e2 = []

  for x in x0:
    eq_ind = np.where(x1==x)
    if len(eq_ind[0]) == 1:
      y2.append(y1[eq_ind[0][0]])
      e2.append(e1[eq_ind[0][0]])
    else:
      min_ind = np.where(x1<x)[0]
      max_ind = np.where(x1>x)[0]
      if len(min_ind) == 0:
        xx1 = x1[max_ind[0]]
        xx2 = x1[max_ind[1]]
        yy1 = y1[max_ind[0]]
        yy2 = y1[max_ind[1]]
        ee1 = e1[max_ind[0]]
        ee2 = e1[max_ind[1]]
        xdet, ydet, edet = lin_interpolate(xx1,yy1,ee1,xx2,yy2,ee2,x)
      elif len(max_ind) == 0:
        xx1 = x1[min_ind[-1]]
        xx2 = x1[min_ind[-2]]
        yy1 = y1[min_ind[-1]]
        yy2 = y1[min_ind[-2]]
        ee1 = e1[min_ind[-1]]
        ee2 = e1[min_ind[-2]]
        xdet, ydet, edet = lin_interpolate(xx1,yy1,ee1,xx2,yy2,ee2,x)
      else:
        xx1 = x1[max_ind[0]]
        yy1 = y1[max_ind[0]]
        ee1 = e1[max_ind[0]]
        xx2 = x1[min_ind[-1]]
        yy2 = y1[min_ind[-1]]
        ee2 = e1[min_ind[-1]]
        xdet, ydet, edet = lin_interpolate(xx1,yy1,ee1,xx2,yy2,ee2,x)

      y2.append(ydet)
      e2.append(edet)

  return x0,y2,e2
      
def subtract(bg, data, f):
  ry = []
  re = [] 

  for ctr, x in enumerate(data[0]):
    ry.append(data[1][ctr]-f*bg[1][ctr])
    re.append(np.sqrt((data[2][ctr]*data[2][ctr])+(f*f*bg[2][ctr]*bg[2][ctr])))
  
  return data[0],ry,re
  
if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Plot one or multiple h5 images.')  

  parser.add_argument('background', nargs=1, type=str, help='Path to where the background file is.',
                      metavar='BACKGROUNDFILES')
  parser.add_argument('path', nargs=1 , type=str, help='Path to the data files.',
                      metavar='PATH')
  parser.add_argument('-o', type=str, help='Name of the output file. Default is nobg.',
                      default='nobg')
  parser.add_argument('-f', type=float, help='Factor to scale bg with. Default is 1.', default=1)
  parser.add_argument('-v', '--verbose', type=bool, help='Show plots during calibration routine.',
                        default=False, const=True, nargs='?')

  args = parser.parse_args()
  #print(args)

  path = args.path[0]
  files = [join(path,f) for f in listdir(path) if (isfile(join(path,f))) and (f.endswith('.dat')) and not (f.endswith('nobg.dat'))]
  bg = args.background[0]
  bgdata = load_data(bg)
  
 
  for fn in files:
    outp = '{}_{}.dat'.format(fn[:-4],args.o) 
    print(outp)
    data = load_data(fn)
    print(data)
    bgspline = spline_data(data[0], bgdata)
    print('debug2')
    
    result = subtract(bgspline, data, args.f)
    result_transp = np.transpose(np.array(result))
    
    if args.verbose:
      plt.loglog(data[0],data[1])
      plt.loglog(bgspline[0],np.multiply(bgspline[1],args.f))
      plt.loglog(result[0], result[1])
      plt.show()
    
    with open(outp, 'w') as of:
      for data in result_transp:
        of.write('{} {} {}\n'.format(data[0],data[1],data[2]))
