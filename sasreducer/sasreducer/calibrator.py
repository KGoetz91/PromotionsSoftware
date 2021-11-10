##############################################################
#Calibrator class handling the actual processing of          #
#2D scattering data.                                         #
#Author: Klaus Götz                                          #
#E-Mail: kgoetz91@t-online.de                                #
#License: GPL-3.0                                            #
#09. November 2021                                           #
##############################################################

import argparse
from distutils.util import strtobool
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt

import pyFAI

import hdf5plugin
import h5py

from scipy.optimize import curve_fit as cfit

from .helper_functions import gaussian
from .masks import Masks

class Calibrator:
  
  def __init__(self):
    pass

  def _integrate_image(self, image, mask, outp):  
    ai = pyFAI.load(self._CALIB)
    #TODO Implement way to put bins into config
    res1000 = ai.integrate1d(image, 1000, unit='q_A^-1', mask = mask, 
                         filename='{}_1000bins.dat'.format(outp), error_model = 'poisson')
    res = ai.integrate1d(image, 100, unit='q_A^-1', mask = mask, 
                         filename='{}_100bins.dat'.format(outp), error_model = 'poisson')
    res = ai.integrate1d(image, 200, unit='q_A^-1', mask = mask,
                         filename='{}_200bins.dat'.format(outp), error_model = 'poisson')
    return res1000

  def _write_poni(self,pixel_size, wavelength):
    """Writes the .poni files needed py the PyFAI library for integration.
    
    pixel_size: size of the detector pixels. Needs to be given in [m] as a string.
    wavelength: wavelength of the used x-rays. Needs to be given in [Angström] as float or string.
    """
    with open(self._CALIB, 'w') as of:
      of.write('#pyFAI Calibration file constructed manually\n')
      of.write('#Created never...\n')
      of.write('poni_version: 2\n')
      of.write('Detector: Detector\n')
      of.write('Detector_config:{{\"pixel1\": {0}, \"pixel2\": {0}, \"max_shape\": null}}\n'.format(pixel_size))
      of.write('Distance: {}\n'.format(self._SDD))
      of.write('Poni1: {}\n'.format(self._COM[0]*float(pixel_size)))
      of.write('Poni2: {}\n'.format(self._COM[1]*float(pixel_size)))
      of.write('Rot1: 0\n')
      of.write('Rot2: 0\n')
      of.write('Rot3: 0\n')
      of.write('Wavelength: {}e-10\n'.format(wavelength))
  

class SAXANSCalibrator(Calibrator):
  
  def _write_poni(self):
    pixel_size = "7.5e-5"
    wavelength = "1.5406"
    super()._write_poni(pixel_size, wavelength)
    
  def _init_calibrator(self, name, params):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-xs', type=int)
    parser.add_argument('-xe', type=int)
    parser.add_argument('-gs', type=int,default=-1)
    parser.add_argument('-ge', type=int,default=-1)
    parser.add_argument('-sdd', type=float)
    parser.add_argument('-cf', type=float)
    parser.add_argument('-p', type=str)
    parser.add_argument('-o', type=str)
    parser.add_argument('-t', type=str)
    parser.add_argument('-v', type=str, default='False', choices=['False', 'True'])
    parser.add_argument('-r', type=int)
    parser.add_argument('-msx', type=int)
    parser.add_argument('-msy', type=int)
    parser.add_argument('-eb', type=int)
    
    args = parser.parse_args(params)
    
    self._NAME = name
    self._XSTART = args.xs
    self._XEND = args.xe
    self._GSTART = args.gs
    self._GEND = args.ge
    self._SDD = args.sdd
    self._SOLID_ANGLE = 75e-6*75e-6/(self._SDD*self._SDD)
    self._CF = args.cf
    self._PATH = args.p
    self._VERBOSE = strtobool(args.v)
    self._INTEGRATION_TYPE = args.t
    self._CALIB = 'temp.poni'
    self._OUTP = args.o
    self._MASK_RADIUS = args.r
    self._MASK_SHIFT_X = args.msx
    self._MASK_SHIFT_Y = args.msy
    self._COM = self._find_direct_beam()
    if args.eb == -1:
      self._EB_FLUX = None
    else:
      self._EB_FLUX = self._calculate_flux(args.eb)
    
  def _calculate_flux(self,number):
    data,xdet,time = self._load_data(number)
    mask = self._create_mask(data)
    return np.sum(np.multiply(data,np.multiply(np.subtract(mask,1),-1)))/time
  
  def _load_data(self, number):
    full_path = join(self._PATH, '{:06d}_master.h5'.format(number))
    print(full_path)
    if not isfile(full_path):
      raise ValueError('The given file {} does not exist.'.format(full_path))
    with h5py.File(full_path) as f:
      wavelength = f['entry/instrument/beam/incident_wavelength'][()]
      time = f['entry/instrument/detector/count_time'][()]
      data = np.zeros((1062,1028))
      for DATAFILE in f['entry/data'].keys():
        new_data = np.array(f['entry/data/{}'.format(DATAFILE)][()])
        frames = new_data.shape[0]
        new_data = np.sum(new_data, 0)
        new_data[new_data == int(np.iinfo(np.uint32).max*frames)] = -1
        new_data = new_data.astype(np.int64)
        data = np.add(data, new_data)
    return data, time
  
  def _find_direct_beam(self):
    xDB = 0
    yDB = 0
    
    image,time = self._load_data(self._XSTART)
    
    if self._VERBOSE:
      plt.imshow(np.log(image))
      plt.show()
      plt.clf()
    
    maxval = np.max(image)
    maxcoord = np.argwhere(image == maxval)
    maxcoord = maxcoord.flatten()
    
    xprojection = np.sum(image,1)
    yprojection = np.sum(image,0)
    maxvalx = np.max(xprojection)
    maxvaly = np.max(yprojection)

    p0 = [maxcoord[0],2,maxvalx]
    x0 = list(range(len(xprojection)))
    p1 = [maxcoord[1],2,maxvaly]
    x1 = list(range(len(yprojection)))
    
    popt0, cov0 = cfit(gaussian, x0, xprojection, p0)
    popt1, cov1 = cfit(gaussian, x1, yprojection, p1)
    
    if self._VERBOSE:
      points = 10000
      x0_plot = np.multiply(np.divide(list(range(points)),points),len(xprojection))
      x1_plot = np.multiply(np.divide(list(range(points)),points),len(yprojection))
      
      plt.plot(xprojection, label='data')
      plt.plot(x0_plot,gaussian(x0_plot,*p0),label='start value')
      plt.plot(x0_plot,gaussian(x0_plot,*popt0), label='fit')
      plt.xlim(maxcoord[0]-20,maxcoord[0]+20)
      plt.legend()
      plt.show()
      plt.plot(yprojection, label='data')
      plt.plot(x1_plot,gaussian(x1_plot,*p1),label='start value')
      plt.plot(x1_plot,gaussian(x1_plot,*popt1), label='fit')
      plt.xlim(maxcoord[1]-20,maxcoord[1]+20)
      plt.legend()
      plt.show()
      
      plt.clf()
      plt.imshow(np.log(image))
      print('{} {}'.format(popt0[0],popt1[0]))
      plt.scatter(popt1[0],popt0[0],c='red', marker='x')
      plt.show()

    xDB = popt0[0]
    yDB = popt1[0]
    

    return (xDB,yDB)
  
  def _calculate_gamma_background(self):
    totalsum = 0 
    totaltime = 0 
    totalpixels = 0 
    result = 0
    if self._GSTART==-1 and self._GEND==-1:
      self._GAMMABG=0
    else:
      if self._GSTART==-1:
        self._GSTART=self._GEND
      elif self._GEND==-1:
        self._GEND=self._GSTART
        
      files = range(self._GSTART, self._GEND+1)

      for number in files:
        data, time = self._load_data(number)
        
        mask = self._create_mask(data)
        totalpixels += len(mask[np.where(mask==0)])
        rev_mask = np.multiply(np.subtract(mask,1),-1)
        data = np.multiply(data, rev_mask)
        
        totaltime += time
        totalsum += np.sum(data)

      result = totalsum/(totaltime*totalpixels)
      self._GAMMABG = result

    if self._VERBOSE:
      print('Sum: {}'.format(totalsum))
      print('Time: {}'.format(totaltime))
      print('Pixels: {}'.format(totalpixels))
      print('Counts per pixel per second: {}'.format(result))
  
  def _create_mask(self, data):
    mask = Masks('saxans', data)
    if self._MASK_RADIUS:
      center = (self._COM[1]+self._MASK_SHIFT_Y,self._COM[0]+self._MASK_SHIFT_X)
      mask = mask.mask_outside_circle(center,self._MASK_RADIUS)
    else:
      mask = mask.mask
    
    return mask
  
  def _integrate_each(self):
    result =[]
    for number in range(self._XSTART,self._XEND+1):
      data,time = self._load_data(number)
      
      mask = self._create_mask(data)
      rev_mask = np.multiply(np.subtract(mask,1),-1)
      
      transmitted_intensity = np.sum(np.multiply(data,rev_mask))
      thickness = np.sqrt(2) #TODO Add a way to define this in config file
      
      data = np.subtract(data, self._GAMMABG*time)
      data = np.divide(data,thickness*transmitted_intensity*self._SOLID_ANGLE)
      data = np.multiply(data,self._CF)
      
      name = '{}_{:08d}'.format(self._NAME,number)
      output = join(self._OUTP,name)
      result.append(self._integrate_image(data,mask,output))
    return result
    
  def _integrate_mean(self):
    result = 0
    sumdata = 0
    totaltime = 0
    
    for number in range(self._XSTART,self._XEND+1):
      data,time = self._load_data(number)
      totaltime += time
      if number == self._XSTART:
        sumdata=data
      else:
        sumdata = np.add(sumdata,data)
    
    mask = self._create_mask(data)
    rev_mask = np.multiply(np.subtract(mask,1),-1)
    
    transmitted_intensity = np.sum(np.multiply(sumdata,rev_mask))
    thickness = np.sqrt(2) #TODO Add a way to define this in config file
    
    sumdata = np.subtract(sumdata, self._GAMMABG*totaltime)
    sumdata = np.divide(sumdata,thickness*transmitted_intensity*self._SOLID_ANGLE)
    sumdata = np.multiply(sumdata,self._CF)
    
    if self._VERBOSE:
      plt.imshow(np.log(sumdata))
      plt.scatter(self._COM[1],self._COM[0],c='red', marker='x')
      plt.show()
      plt.clf()
      plt.imshow(np.log(np.multiply(sumdata,rev_mask)))
      plt.show()
    
    name = '{}_mean'.format(self._NAME)
    output = join(self._OUTP, name)
    result = self._integrate_image(sumdata,mask,output)
    
  def __init__(self, name, params):
    super().__init__()
    self._init_calibrator(name, params)
    self._calculate_gamma_background()
    self._write_poni()
    
  def integrate(self):
    self.print_info()
    if self._INTEGRATION_TYPE == 'mean':
      result = self._integrate_mean()
    elif self._INTEGRATION_TYPE == 'each':
      result = self._integrate_each()
      
    return result
  
  def print_info(self):
    text = 'Reducing sample {}:\n'.format(self._NAME)
    if self._EB_FLUX:
      text += 'Transmission: {}'.format(self._calculate_flux(self._XSTART)/self._EB_FLUX)
    else:
      text += 'No empty beam, Transmission calculation not possible.'
    
    print(text)
