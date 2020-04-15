#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.rcParams as pltParams
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit as fit

def line(x, m, t):
    return np.multiply(x,m)+t

def load_data(fn):
  result = []
  with open(fn, 'r') as f:
    for line in f:
      data = line.split()
      result.append(data)
  
  return np.transpose(np.array(result,dtype=float))

def load_TEM_data(fn):
  
  diams = []
  
  with open('{}'.format(fn), 'r') as f:
    for ctr,line in enumerate(f):
      if ctr > 0:
        #nr, area, major, minor, angle = line.split(',')
        data = line.split(',')
        major = data[2]
        minor = data[3]
        diams.append((float(major)+float(minor))/2)
  
  return diams

def load_sasfit_data(fn):
  result = {}
  with open(fn,'r') as f:
    for ctr,line in enumerate(f):
      data = line.strip().split(';')
      if ctr == 0:
        for i in range(len(data)):
          result[i]=[]
      for i in range(len(data)):
        try:
          result[i].append(float(data[i]))
        except:
          pass
      
  return (result)
  #return np.transpose(np.array(result,dtype=float))

def loadRAS(fn):
  datasets = [] 
  with open(fn, 'r', encoding='latin_1') as f:
    for line in f:
      if line.startswith('*'):
        if line.startswith('RAS_INT_START', 1):
          datasets.append({})
      else:
        x,y,e = line.split()
        if 'data' in datasets[-1]:
          datasets[-1]['x_val'] = np.append(datasets[-1]['x_val'], float(x))
          datasets[-1]['data'] = np.append(datasets[-1]['data'], float(y))
          datasets[-1]['errors'] = np.append(datasets[-1]['errors'], float(e))
        else:
          datasets[-1]['x_val'] = np.array(float(x))
          datasets[-1]['data'] = np.array(float(y))
          datasets[-1]['errors'] = np.array(float(e))

  return datasets

def fit_line(starting_values, xval, yval):
    fitfunc = lambda x, m:line(x,m,0) 
    fit_result = fit(fitfunc, xval, yval, starting_values)
    print(fit_result)
    return fit_result


if __name__ == '__main__':
  
  fn = sys.argv[1]
  hist = sys.argv[2]
  
  data = load_sasfit_data(fn)
  #linefit = fit_line([2], data[0], data[1])
  #qval = data[0].get('x_val')
  #qval = np.multiply(qval,(np.pi)/(360))
  #qval = np.multiply(np.sin(qval),(4*np.pi)/0.15406)
  diams = load_TEM_data(hist)
  diams = np.divide(diams,2)
  inten = np.multiply(data[1], np.power(data[0],3))
  inten = np.multiply(np.divide(inten, np.amax(inten)),38)
  
  
  #plt.style.use('ggplot')
  plt.rcParams.update({'font.size':30})
  plt.rcParams.update({'axes.linewidth':8})
  #plt.rcParams.update({'ticks.linewidth':8})
  figure, ax = plt.subplots(figsize=(14,14))
  
  ax.tick_params(length=16, width = 8, pad=10)
  figure.subplots_adjust(left=0.2)
  
  #print(data[6])
  #print(data[7])
  
  #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  plt.xlabel(r'R (nm)')
  plt.ylabel(r'N(R)*R$^3$ (a.u.)')
  #print(data[0])
  #print(data[1])
  plt.plot(data[0], inten, label='Particle Core Radius', c='red')
  #plt.plot(data[0], line(data[0], linefit[0][0],0), c = 'grey')
  plt.hist(diams, bins=10, label='TEM Histogram')
  #plt.loglog(data[4],data[5], 'ro', label='SAXS Data')
  #plt.loglog(data[8],data[9], label='SANS Fit', c='blue')
  #plt.loglog(data[12],data[13], 'bo', label='SANS Data')
  leg = plt.legend(markerscale = 3.)
  for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)
  plt.xlim(0,10) 
  #plt.ylim(400,1600)
  
  #plt.show()
  plt.savefig('{}.png'.format(fn[:-4]))
