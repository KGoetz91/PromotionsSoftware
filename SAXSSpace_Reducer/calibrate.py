#!/usr/bin/python3

from PIL import Image

from scipy.optimize import curve_fit as cfit

import pyFAI

from os import listdir
from os.path import join, isfile
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage 

import argparse

class DataSet:
  
  def _load_data(self, fn):
    files = [join(fn,f) for f in listdir(fn) 
              if (isfile(join(fn,f))) and (f.endswith('.tif')) and not(f.endswith('corrected.tif'))]

    if len(files) > 0:
      shape = np.array(Image.open(files[0])).shape
      result = np.zeros(shape)
      
      for data in files:
        im = Image.open(data)
        result = np.add(result, np.array(im))

      if self._VERBOSE:
        plt.imshow((result))
        plt.imshow(np.log(result))
        plt.show()
        
    else:
      raise ValueError('Path to data is empty.')
        
    return result
    
  def _gauss(self, x, x0, sigma, maximum):
    dx = np.subtract(x,x0)
    quotient = -2*sigma**2
    exponent = np.divide(np.power(dx,2),quotient)
    return maximum*np.exp(exponent)

  def _find_direct_beam(self):

    xDB = 0
    yDB = 0
    
    maxval = np.max(self._twoddata)
    maxcoord = np.where(self._twoddata == maxval)
    
    xprojection = np.sum(self._twoddata,1)
    yprojection = np.sum(self._twoddata,0)
    maxvalx = np.max(xprojection)
    maxvaly = np.max(yprojection)

    p0 = [maxcoord[0][0],2,maxvalx]
    x0 = list(range(len(xprojection)))
    p1 = [maxcoord[1][0],2,maxvaly]
    x1 = list(range(len(yprojection)))
    
    popt0, cov0 = cfit(f=self._gauss, xdata=x0, ydata=xprojection, p0=p0)
    popt1, cov1 = cfit(self._gauss, x1, yprojection, p1)
    
    if self._VERBOSE:
      points = 10000
      x0_plot = np.multiply(np.divide(list(range(points)),points),len(xprojection))
      x1_plot = np.multiply(np.divide(list(range(points)),points),len(yprojection))
      
      plt.plot(xprojection)
      plt.plot(x0_plot,self._gauss(x0_plot,*p0))
      plt.plot(x0_plot,self._gauss(x0_plot,*popt0))
      plt.xlim(maxcoord[0]-20,maxcoord[0]+20)
      plt.show()
      plt.plot(yprojection)
      plt.plot(x1_plot,self._gauss(x1_plot,*p1))
      plt.plot(x1_plot,self._gauss(x1_plot,*popt1))
      plt.xlim(maxcoord[1]-20,maxcoord[1]+20)
      plt.show()

    xDB = popt0[0]
    yDB = popt1[0]

    return (xDB,yDB)
  
  def _create_mask(self):
    mask = np.array(self._twoddata)
    #y = 388
    #x = 398
    #print(mask[x,y])
    #print(mask[x-2:x+2,y-2:y+2])
    mask[mask >= 0] = 0
    mask[mask < 0] = 1
    mask[300,409] = 1
    mask[398,388] = 1
    #mask[666,388] = 1
    #mask[764,409] = 1
    #mask[1006,274] = 1
    mask[:int(self._com[0])]=1
    #np.save('mask.npy', mask)
    if self._VERBOSE:
      plt.imshow(np.log10(np.multiply(self._twoddata,np.multiply(-1,np.subtract(mask,1)))))
      plt.show()
    return mask
  
  def _write_poni(self, distance):
    with open(self._CALIB, 'w') as of:
      of.write('#pyFAI Calibration file constructed manually\n')
      of.write('#Created never...\n')
      of.write('poni_version: 2\n')
      of.write('Detector: Detector\n')
      of.write('Detector_config: {"pixel1": 7.5e-5, "pixel2": 7.5e-5, "max_shape": null}\n')
      of.write('Distance: {}\n'.format(distance))
      of.write('Poni1: {}\n'.format(self._com[0]*7.5e-5))
      of.write('Poni2: {}\n'.format(self._com[1]*7.5e-5))
      of.write('Rot1: 0\n')
      of.write('Rot2: 0\n')
      of.write('Rot3: 0\n')
      of.write('Wavelength: 1.5406e-10\n')

  def _integrateoned(self,distance, outp):
    self._write_poni(distance)
    ai = pyFAI.load(self._CALIB)
    res1000 = ai.integrate1d(self._twoddata, 1000, unit='q_nm^-1', mask = self._mask, 
                         filename='{}_1000bins.dat'.format(outp), error_model = 'poisson')
    res = ai.integrate1d(self._twoddata, 100, unit='q_nm^-1', mask = self._mask, 
                         filename='{}_100bins.dat'.format(outp), error_model = 'poisson')
    res = ai.integrate1d(self._twoddata, 200, unit='q_nm^-1', mask = self._mask,
                         filename='{}_200bins.dat'.format(outp), error_model = 'poisson')
    return res1000

  def __init__(self, params):
    self._CALIB = 'temp.poni'
    self._VERBOSE = params['verbose']
    
    self._paramlist = ['cf', 'thickness', 'path', 'distance', 'output']
    for i in self._paramlist:
      if not i in params:
        raise ValueError('Parmeters need the following components: {}'.format(self._paramlist)) 
    
    self._twoddata = self._load_data(params['path'])
    
    self._com = self._find_direct_beam() 
    self._mask = self._create_mask() 
    
    self._imagesum = np.sum(np.multiply(self._twoddata,np.multiply(-1,np.subtract(self._mask,1))))
    self._twoddata = np.multiply(self._twoddata, params['cf']/(self._imagesum*params['thickness']))
    
    self._oneddata = self._integrateoned(params['distance'], params['output'])
    
  def image(self):
    return self._twoddata
  
  def curve(self):
    return self._oneddata
  
  def plot_image(self):
    plt.imshow(self._twoddata)
    plt.show()
    
  def plot_curve(self):
    plt.plot(self._oneddata[0], self._oneddata[1])
    plt.show()
    
  def plot_logcurve(self):
    plt.loglog(self._oneddata[0], self._oneddata[1])
    plt.show()

class Calibrator:
  __DIST__ = 0.3075
  __CF__ = 14.5e7
 
  def _init_parameters(self, params):
    parser = argparse.ArgumentParser(description=
                                    'Integrate one or more 2D images and calibrate the data absolute.')

    parser.add_argument('path', nargs=1 , type=str, help='Path to the files.',
                        metavar='PATH')
    parser.add_argument('inputfiles', nargs='+', type=str, help='Path to where the files are.',
                        metavar='INPUTFILES')
    parser.add_argument('-cf', type=float, help='Calibration factor. The data will be multiplied by this.',
                        default=self.__CF__)
    parser.add_argument('-d', type=float, help='Sample-Detector distance. Default is 0.3075 m.',
                        default=self.__DIST__)
    parser.add_argument('-o', type=str, help='Name of the output file. Default is integrated.',
                        default='integrated')
    parser.add_argument('-t', type=float, help='Measurement time. Default is 1 second.',
                        default=1)  
    parser.add_argument('-s', type=float, help='Sample thickness. Default is 1 mm.',
                        default=1)
    parser.add_argument('-i0', type=float, 
                        help='Flux of the incoming beam in counts per second. Default is 1.', default=2.54e6)
    parser.add_argument('-v', '--verbose', type=bool, help='Show plots during calibration routine.',
                        default=False, const=True, nargs='?')
    parser.add_argument('-p', '--param', type=bool, help='Use a parameter file for INPUTFILES.',
                        default=False, const=True, nargs='?')

    args = parser.parse_args(params)
    print(args)

    self._PATH = args.path[0]
    self._FILES = args.inputfiles
    self._CF = args.cf
    self._TIME = args.t
    self._THICK = args.s
    self._DIST = args.d 
    self._I0 = args.i0
    self._VERBOSE = args.verbose
    self._PARAM = args.param
    self._OUTPUT = args.o
    
  def _load_data(self): 
    if self._PARAM:
      raise NotImplementedError('This function is not implemented yet.')
    else:
      for fn in self._FILES:
        params = {}
        params['verbose'] = self._VERBOSE
        params['path'] = join(self._PATH, fn) 
        params['distance'] = self._DIST 
        params['output'] = join(self._PATH, fn, self._OUTPUT) 
        params['cf'] = self._CF
        params['thickness'] = self._THICK
        self._datasets.append(DataSet(params))
 
  def __init__(self, params):
    self._init_parameters(params)
    self._datasets = []
    self._load_data()

  def work(self):
    for d in self._datasets:
      d.plot_logcurve()


if __name__ == '__main__':

  c = Calibrator(sys.argv[1:])
  c.work()
