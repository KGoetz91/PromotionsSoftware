#!/usr/bin/python3.8
"""This module handles loading images from the SAXS/SANS setup at D22 ILL"""

import hdf5plugin
import h5py

import numpy as np

import argparse

from os.path import join, isfile
from os import listdir

class SaxansLoader:
  
  def load_image(self, full_path):
    with h5py.File(full_path) as f:
      number = list(list(full_path.split('/'))[-1].split('.'))[0]
      print(number)
      data = f['{}/data/image'.format(number)][()]
      monitor = np.sum(data[data>0])
      time = float(f['{}/instrument/det/preset'.format(number)][()])
      thickness = np.sqrt(2)
      xdet = float(f['{}/instrument/xdet/value'.format(number)][()])
      if xdet < 10:
        distance = self._SHORTDIST
        cf = self._CF_SD
      elif xdet < 200:
        distance = self._MEDIUMDIST
        cf = self._CF_MD
      else:
        distance = self._LARGEDIST
        cf = self._CF_LD
    params = {'monitor':monitor, 'time': time, 'thickness':thickness,
              'distance': distance, 'cf':cf, 'xdet':xdet}
  
    return data, params
    
  def load_set(self, path, file_range):
    files = []
    
    for filenumber in file_range:
      files.append(join(path,'{:08d}.h5'.format(filenumber)))

    result = {}
    for data_file in files:
      number = list(list(data_file.split('/'))[-1].split('.'))[0]
      print(number)
      result[number]=(self.load_image(data_file))
      
    return result
  
  def _set_class_variables(self, args):
    parser = argparse.ArgumentParser(description='This class loads SAXANS h5 files.')
    parser.add_argument('-cfl', '--cflong', type = float, help='Calibration factor for long distances.',
                        default=1)
    parser.add_argument('-cfm', '--cfmedium', type = float, help='Calibration factor for medium distances.',
                        default=1)
    parser.add_argument('-cfs', '--cfshort', type = float, help='Calibration factor for short distances.',
                        default=1)
    parser.add_argument('-ld', '--longdistance', type = float, help='Long sample detector distance.',
                        default=1.5813)
    parser.add_argument('-md', '--mediumdistance', type = float, help='Long sample detector distance.',
                        default=0.6347)
    parser.add_argument('-sd', '--shortdistance', type = float, help='Long sample detector distance.',
                        default=0.53876)
    
    params = parser.parse_args(args)
    
    self._CF_LD = params.cflong
    self._CF_MD = params.cfmedium
    self._CF_SD = params.cfshort
    self._LARGEDIST = params.longdistance
    self._MEDIUMDIST = params.mediumdistance
    self._SHORTDIST = params.shortdistance
  
  def __init__(self, args=[]):
    self._set_class_variables(args)
