#!/usr/bin/python3
"""This module handles all operations performed on the 2D image."""

import numpy 

import argparse

class DataSet:
 
  
 
  def _set_class_variables(self, args):
    parser = argparse.ArgumentParser(description='This class loads SAXANS h5 files.')
    
    parser.add_argument('-t', '--time', type=float, help ='Measurement time.', default=1)
    parser.add_argument('-m', '--monitor', type=float, help ='Monitor value of image.', default=1)
    parser.add_argument('-t', '--thickness', type=float, help ='Sample thickness.', default=1)
    parser.add_argument('-d', '--distance', type=float, help ='Sample detector distance.', default=1)
    parser.add_argument('-c', '--cf', type=float, help ='Calibration factor.', default=1)
    
    params = parser.parse_args(args)
    
    if type(data) == numpy.ndarray:
      self._data = data
    else:
      raise TypeError('Data needs to be a numpy array.')
    self._time = params.time
    self._monitor = params.monitor
    self._thickness = params.thickness
    self._distance = params.distance
    self._cf = params.cf
  
  def __init__(self, data, args=[]):
    if type(args) == dict:
      newargs = []
      for key in args.keys():
        newargs.append('--{}'.format(key))
        newargs.append('{}'.format(args[key]))
      args = newargs
    self._set_class_variables(data, args)