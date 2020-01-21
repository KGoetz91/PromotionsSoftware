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
  wlctr = 1000
  with open(fn, 'r') as f:
    for ctr, line in enumerate(f):
      data = line.strip().split()
      if data[0] == 'nm':
        wl = data[1:]
        wl = [float(x.replace(',','.')) for x in wl]
        wlctr = ctr
      if ctr > wlctr:
        time = float(data[0])
        intensity = data[1:]
        intensity = [float(x.replace(',','.')) for x in intensity]
        
        result[time] = {'x': wl, 'y': intensity}
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
    print(i)
    result = load_uvVis('{}_{:03d}.txt'.format(path,i))
    for key in result.keys():
      data.append(result[key])
  
  outp = open('{}_{}-{}_fitResult_AllPeak.dat'.format(path, firstnumber, lastnumber), 'w')
  #outp = open('{}_{}-{}_fitResult_LargePeak.dat'.format(path, firstnumber, lastnumber), 'w')
  #outp = open('{}_{}-{}_fitResult_SmallPeak.dat'.format(path, firstnumber, lastnumber), 'w')
  
  plt.style.use('ggplot')
  plt.rcParams.update({'font.size':22})
  figure, ax = plt.subplots(figsize=(14,8))
  

  
  for dset in data:
    plt.clf()
    plt.plot(dset['x'], dset['y'])
    #plt.show()
    #plt.xlim([380,640])
    #plt.ylim([-0.02,0.3])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorbance [AU]')
    starting_values1 = [420, 10, 30, -0.0005, 0.26] #large Peak
    starting_values2 = [550, 10, 30, -0.0005, 0.26] #medium peak
    starting_values3 = [585, 10, 30, -0.0005, 0.26] #small peak
    x_fit2 = [x if x >= 490 and x <= 620 else 0 for x in dset['x']]  # Medium Peak
    x_fit1 = [x if x >= 350 and x <= 490 else 0 for x in dset['x']] # Large Peak
    x_fit3 = [x if x >= 570 and x <= 650 else 0 for x in dset['x']]  # Small Peak
    y_fit1=np.array(dset['y'])[np.nonzero(x_fit1)]
    y_fit2=np.array(dset['y'])[np.nonzero(x_fit2)]
    y_fit3=np.array(dset['y'])[np.nonzero(x_fit3)]
    x_fit1=np.array(x_fit1)[np.nonzero(x_fit1)]
    x_fit2=np.array(x_fit2)[np.nonzero(x_fit2)]
    x_fit3=np.array(x_fit3)[np.nonzero(x_fit3)]
    fit_result1 = fit(fitfunc, x_fit1, y_fit1, starting_values1)
    fit_result2 = fit(fitfunc, x_fit2, y_fit2, starting_values2)
    fit_result3 = fit(fitfunc, x_fit3, y_fit3, starting_values3)
    outp.write('Fläche Gauss LPeak: {}\n'.format(fit_result1[0][2]))
    outp.write('Fläche Gauss MPeak: {}\n'.format(fit_result2[0][2]))
    outp.write('Fläche Gauss SPeak: {}\n'.format(fit_result3[0][2]))
    #print('Fläche Gauss: {}'.format(fit_result[0][2]))
    plt.plot(x_fit1, fitfunc(x_fit1, *fit_result1[0]))
    plt.plot(x_fit2, fitfunc(x_fit2, *fit_result2[0]))
    plt.plot(x_fit3, fitfunc(x_fit3, *fit_result3[0]))
    #plt.plot(x_fit, fitfunc(x_fit, *starting_values))
    if mode == 'v':
      #plt.savefig('outp.png')
      plt.show()
  
  outp.close()