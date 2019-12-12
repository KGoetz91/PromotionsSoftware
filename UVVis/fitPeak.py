#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit
from math import pi

def gauss(x, x0, sigma, amplitude):
    
    return (amplitude/(sigma*np.sqrt(2*pi)))*np.exp(-0.5*np.power(((np.subtract(x,x0))/sigma),2))

def line(x, m, t):
    return np.multiply(x,m)+t

fitfunc = lambda x, x0, sigma, amplitude, m, t : gauss(x, x0, sigma, amplitude)+line(x,m,t)

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
  return result

if __name__ == '__main__':

  firstnumber = int(sys.argv[1])
  lastnumber = int(sys.argv[2])
  files = range(firstnumber,lastnumber+1, 1)
  data = []
  for i in files:
    result = load_uvVis('000_{:03d}.dat'.format(i))
    for key in result.keys():
      data.append(result[key])
  
  for dset in data:
    plt.clf()
    starting_values = [430, 10, 30, -0.0005, 0.26]
    x_fit = [x if x >= 350 and x <= 490 else 0 for x in dset['x']]
    y_fit=np.array(dset['y'])[np.nonzero(x_fit)]
    x_fit=np.array(x_fit)[np.nonzero(x_fit)]
    fit_result = fit(fitfunc, x_fit, y_fit, starting_values)
    print('Fläche Gauss: {}'.format(fit_result[0][2]))
    plt.plot(dset['x'], dset['y'])
    plt.plot(x_fit, fitfunc(x_fit, *fit_result[0]))
    #plt.plot(x_fit, fitfunc(x_fit, *starting_values))
    plt.show()
