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

class Masks:
  def _saxans(self, data):
    mask = np.array(data)
    mask[mask >= 0] = 0 
    mask[mask < 0] = 1 
    mask[666,388] = 1 
    mask[764,409] = 1 
    mask[1006,274] = 1 
    
    return mask

  def _saxspace(self, data):
    mask = np.array(self._twoddata)
    #y = 388
    #x = 398
    #print(mask[x,y])
    #print(mask[x-2:x+2,y-2:y+2])
    mask[mask >= 0] = 0
    mask[mask < 0] = 1
    mask2 = np.array(mask)
    
    radius = 30
    alpha = 45
    coordinates = []
    for x in range(-radius, radius, 1):
      for y in range(-radius, radius, 1):
        if x*x+y*y<=radius*radius:
          if (x>0) or (abs(y) > abs(x*np.tan(alpha*np.pi/180))):
            coordinates.append((x,y))
        
    coordinates = [(x+int(self._com[0])+1,y+int(self._com[1])) for (x,y) in coordinates]
    for (x,y) in coordinates:
      mask2[x,y] = 1000
      
    #plt.imshow(mask2)
    plt.imshow(np.log10(np.multiply(self._twoddata,np.multiply(-1,np.subtract(mask2,1)))))
    plt.show()
    
    #mask2[int(self._com[0])-10:int(self._com[0])+20, int(self._com[1])-10: int(self._com[1])+20] = 1
    #data_temp = np.multiply(self._twoddata, np.multiply(-1,np.subtract(mask2,1)))
    #mask[300,409] = 1 #mask[398,388] = 1 # Works only for my PC at home
    mask[666,388] = 1 
    mask[764,409] = 1 
    mask[1006,274] = 1 
    #mask[:int(self._com[0])]=1 # Works only for my PC at home
    mask[int(self._com[0])+1:]=1
    #np.save('mask.npy', mask)
    if self._VERBOSE:
      plt.imshow(np.log10(np.multiply(self._twoddata,np.multiply(-1,np.subtract(mask,1)))))
      plt.show()
    return mask
  
  def __init__(self, mask_type):
    self.supported_types = {'saxans': self._saxans,'saxspace': self._saxspace}
    if not(mask_type in supported_types.keys()):
      error_message = 'Supported mask types are:'
      for t in self.supported_types.keys:
        error_message += ' {}'.format(t)
      error_message += '.'
      raise ValueError(error_message)
    self.mask = self.supported_types[mask_type]

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
    mask2 = np.array(mask)
    
    radius = 30
    alpha = 45
    coordinates = []
    for x in range(-radius, radius, 1):
      for y in range(-radius, radius, 1):
        if x*x+y*y<=radius*radius:
          if (x>0) or (abs(y) > abs(x*np.tan(alpha*np.pi/180))):
            coordinates.append((x,y))
        
    coordinates = [(x+int(self._com[0])+1,y+int(self._com[1])) for (x,y) in coordinates]
    for (x,y) in coordinates:
      mask2[x,y] = 1000
      
    #plt.imshow(mask2)
    plt.imshow(np.log10(np.multiply(self._twoddata,np.multiply(-1,np.subtract(mask2,1)))))
    plt.show()
    
    #mask2[int(self._com[0])-10:int(self._com[0])+20, int(self._com[1])-10: int(self._com[1])+20] = 1
    #data_temp = np.multiply(self._twoddata, np.multiply(-1,np.subtract(mask2,1)))
    #mask[300,409] = 1 #mask[398,388] = 1 # Works only for my PC at home
    mask[666,388] = 1 
    mask[764,409] = 1 
    mask[1006,274] = 1 
    #mask[:int(self._com[0])]=1 # Works only for my PC at home
    mask[int(self._com[0])+1:]=1
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

class Workload:
  def _init_setups(self, setup_file):
    _setups = {}
    if not(isfile(setup_file)):
      raise ValueError('The chosen setup file does not exist.')
    else:
      _qualifiers = ['--setup-name', '--sample-detector-distance', '--calibration-factor', '--file-path']
      with open(setup_file, 'r') as f:
        for line in f:
          if not(line.startswith('#')):
            if line.startswith('!'):
              line = line[1:]
              _qualifiers = list(line.strip().split(','))
            else:
              _setups.update(self._parse_setup(_qualifiers, line))
    return _setups
              
  def _parse_setup(self, qualifiers, line):
    line = list(line.strip().split(','))
    parsing_args = np.array(list(zip(qualifiers,line))).flatten()
    
    parser = argparse.ArgumentParser(description='Parses the setup inputs.')
    
    parser.add_argument('--setup-name', type=str, default ='placeholder setup')
    parser.add_argument('--sample-detector-distance', type=float, default = 1)
    parser.add_argument('--calibration-factor', type=float, default=1)
    parser.add_argument('--file-path', type=str, default='./')
    
    args = parser.parse_args(parsing_args)
    
    return {args.setup_name: ['-sdd', '{}'.format(args.sample_detector_distance), '-cf', '{}'.format(args.calibration_factor), '-p', args.file_path]}
  
  def _init_workload(self, data_file):
    _workload = {}
    if not(isfile(data_file)):
      raise ValueError('The chosen data file does not exist.')
    else:
      _qualifiers=['--sample-name', '--xdet', '--xstart', '--xend', '--gamma-start', '--gamma-end']
      with open(data_file, 'r') as f:
        for line in f:
          if not(line.startswith('#')):
            if line.startswith('!'):
              line = line[1:]
              _qualifiers = list(line.strip().split(','))
            else:
              new_workload = self._parse_workload(_qualifiers,line)
              if not (0 in new_workload.keys()):
                new_sample = list(new_workload.keys())[0]
                if new_sample in _workload.keys():
                  new_sample_name = self._iterate_sample_name(_workload.keys(), new_sample)
                  _workload.update({new_sample_name: new_workload[new_sample]})
                else:
                  _workload.update(new_workload)
    return _workload
  
  def _iterate_sample_name(self, keys, sample_name):
    new_sample_name = sample_name
    while new_sample_name in keys:
      name_elements = new_sample_name.split('_')
      number = int(name_elements[-1])
      new_sample_name = '{}_{}_{:04d}'.format(name_elements[0], name_elements[1], number+1)
    return new_sample_name
  
  def _parse_workload(self,qualifiers,line):
    line = list(line.strip().split(','))
    parsing_args = np.array(list(zip(qualifiers,line))).flatten()
    
    parser = argparse.ArgumentParser(description='Parses the data file input.')
    
    parser.add_argument('--sample-name', type=str, default='no-name-given')
    parser.add_argument('--setup-name', type=str, default='no-name-given')
    parser.add_argument('--xdet', type=float, default = -1)
    parser.add_argument('--xstart', type=int, default = -1)
    parser.add_argument('--xend', type=int, default = -1)
    parser.add_argument('--gamma-start', type=int, default = -1)
    parser.add_argument('--gamma-end', type=int, default = -1)
    
    args = parser.parse_args(parsing_args)
    
    if (args.xstart == -1 and args.xend == -1) or (args.xdet == -1):
      print('Not enough input, skipping line: {}'.format(line))
      return {0:0}
    if (args.sample_name == 'no-name-given'):
      if args.xstart == -1:
        args.sample_name = '{}'.format(args.xend)
      else:
        args.sample_name = '{}'.format(args.xstart)
    if (args.setup_name == 'no-name-given'):
      args.setup_name = '{}'.format(int(args.xdet))
    if args.xstart == -1:
      args.xstart = args.xend
    if args.xend == -1:
      args.xend = args.xstart
    
    setup = self._setups[args.setup_name]
    result = ['-xs', '{}'.format(args.xstart),
              '-xe', '{}'.format(args.xend),] + setup
    if not(args.gamma_start == -1 and args.gamma_end == -1):
      if args.gamma_start == -1:
        args.gamma_start == args.gamma_end
      elif args.gamma_end ==-1:
        args.gamma_end = args.gamma_start
      result = result + ['-gs', '{}'.format(args.gamma_start),
                         '-ge', '{}'.format(args.gamma_end)]
      
      return {'{}_{}_0001'.format(args.sample_name, args.setup_name): result}
      
  def __init__(self, setup_file, data_file):
    self._setups = self._init_setups(setup_file)
    self._workload = self._init_workload(data_file)
    
  def setups(self):
    return self._setups

  def workload(self):
    return self._workload
    
class SAXSSpaceCalibrator:
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

class SAXANSCalibrator:
  
  def __init__(self):
    pass

class Worker:
  
  def _init_parameters(self, params):
    
    self._ALLOWED_TYPES = {'saxans':SAXANSCalibrator,
                           'saxspace':SAXSSpaceCalibrator}
    parser = argparse.ArgumentParser()
    
    parser.add_argument('setup', type=str, help='File containing the setups that will be loaded.')
    parser.add_argument('data', type=str, help='File containing the formatted data information.')
    parser.add_argument('-v', '--verbose', type=bool, help='Display steps along the way.', default=False, const=True, nargs='?')
    parser.add_argument('-t', '--type', type=str,
                        help='Which SAXS data is to be refined.', 
                        default = 'saxans',
                        choices=list(self._ALLOWED_TYPES.keys()))
    parser.add_argument('-o', '--output', type=str, default = './',
                        help='Path were the outputs will be saved.')
    
    args = parser.parse_args(params)
    self._SETUP = args.setup
    self._DATA = args.data
    self._VERBOSE = args.verbose
    self._TYPE = args.type
    self._OUTP = args.output
  
  def __init__(self, params):
    self._init_parameters(params)
    self.workload = Workload(self._SETUP, self._DATA)
    
    if self._VERBOSE:
      print(self.workload.setups())
      print(self.workload.workload())

if __name__ == '__main__':

  c = Worker(sys.argv[1:])
