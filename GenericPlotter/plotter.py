#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

if __name__ == '__main__':
  
  fn = sys.argv[1]
  hist = sys.argv[2]
  
  data = load_sasfit_data(fn)
  diams = load_TEM_data(hist)
  diams = np.divide(diams,2)
  inten = np.multiply(data[1], np.power(data[0],3))
  inten = np.multiply(np.divide(inten, np.amax(inten)),38)
  
  
  plt.style.use('ggplot')
  plt.rcParams.update({'font.size':22})
  figure, ax = plt.subplots(figsize=(14,8))
  
  #print(data[6])
  #print(data[7])
  
  #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  plt.ylabel(r'N(R)*R$^3$ [a.u.]')
  plt.xlabel(r'R [nm]')
  plt.plot(data[0],inten, label='Particle Core Radius')
  plt.hist(diams, bins=10, label='TEM Histogram')
  #plt.loglog(data[4],data[5], 'ro', label='SAXS Data')
  #plt.loglog(data[8],data[9], label='SANS Fit')
  #plt.loglog(data[12],data[13], 'bo', label='SANS Data')
  plt.legend()
  plt.xlim(0,10) 
  #plt.ylim(400,1600)
  
  #plt.show()
  plt.savefig('{}.png'.format(fn[:-4]))
