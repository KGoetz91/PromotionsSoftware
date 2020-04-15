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
  
  path = sys.argv[1]
  firstnumber = int(sys.argv[2])
  lastnumber = int(sys.argv[3])
  if len(sys.argv) >= 5:
    mode = sys.argv[4]
  else:
    mode = 'v'
    
  files = range(firstnumber,lastnumber+1, 1)
  data = []
  for i in files:
    result = load_uvVis('{}_{:03d}.dat'.format(path,i))
    for key in result.keys():
      data.append(result[key])
  
  outp = open('{}_{}-{}_fitResult_SmallPeak.dat'.format(path, firstnumber, lastnumber), 'w')
  #outp = open('{}_{}-{}_fitResult_mediumPeak.dat'.format(path, firstnumber, lastnumber), 'w')
  #outp = open('{}_{}-{}_fitResult_LargePeak.dat'.format(path, firstnumber, lastnumber), 'w')
  
  for dset in data:
    plt.clf()
    #starting_values = [420, 10, 30, -0.0005, 0.26] #large peak
    #starting_values = [545, 10, 30, -0.0005, 0.26] #medium peak
    starting_values = [585, 10, 30, -0.0005, 0.26] #small peak
    #x_fit = [x if x >= 490 and x <= 620 else 0 for x in dset['x']]  # Medium Peak
    #x_fit = [x if x >= 350 and x <= 490 else 0 for x in dset['x']] # Large Peak
    x_fit = [x if x >= 570 and x <= 650 else 0 for x in dset['x']]  # Small Peak
    y_fit=np.array(dset['y'])[np.nonzero(x_fit)]
    x_fit=np.array(x_fit)[np.nonzero(x_fit)]
    fit_result = fit(fitfunc, x_fit, y_fit, starting_values)
    outp.write('Fläche Gauss: {}\n'.format(fit_result[0][2]))
    print('Fläche Gauss: {}'.format(fit_result[0][2]))
    plt.plot(dset['x'], dset['y'])
    plt.plot(x_fit, fitfunc(x_fit, *fit_result[0]))
    #plt.plot(x_fit, fitfunc(x_fit, *starting_values))
    if mode == 'v':
      plt.show()
  
  outp.close()