#

from os.path import join, isfile
import numpy as np
import argparse

class Workload:
  def _init_setups(self, setup_file):
    _setups = {}
    if not(isfile(setup_file)):
      raise ValueError('The chosen setup file does not exist.')
    else:
      _qualifiers = ['--setup-name', '--sample-detector-distance', '--calibration-factor', '--file-path', '--mask-radius', '--mask-shift-y', '--mask-shift-x']
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
    parser.add_argument('--mask-radius', type=int)
    parser.add_argument('--mask-shift-x', type=int, default=0)
    parser.add_argument('--mask-shift-y', type=int, default=0)
    parser.add_argument('--empty-beam', type=int, default=-1)
      
    args = parser.parse_args(parsing_args)
    
    result = ['-sdd', '{}'.format(args.sample_detector_distance), '-cf', '{}'.format(args.calibration_factor), '-p', args.file_path, '-msx', '{}'.format(args.mask_shift_x), '-msy', '{}'.format(args.mask_shift_y), '-eb', '{}'.format(args.empty_beam)]
    if args.mask_radius:
      result = result+['-r', '{}'.format(args.mask_radius)]
    return {args.setup_name: result}
  
  def _init_workload(self, data_file):
    _workload = {}
    if not(isfile(data_file)):
      raise ValueError('The chosen data file does not exist.')
    else:
      _qualifiers=['--sample-name', '--xdet', '--xstart', '--xend', '--gamma-start', '--gamma-end', '--integration-type']
      with open(data_file, 'r', encoding ='utf-8-sig') as f:
        for line in f:
          if not(line.startswith('#')) :
            print(line)
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
  
  def _clean_parsing_args(self,args):
    args = list(args)
    while('----' in args):
      position = args.index('----')
      args.pop(position)
      args.pop(position-1)
    while('' in args):
      position = args.index('')
      args.pop(position)
      args.pop(position-1)
    while('-' in args):
      position = args.index('-')
      args.pop(position)
      args.pop(position-1)
      
    return np.array(args)
  
  def _parse_workload(self,qualifiers,line):
    line = list(line.strip().split(','))
    parsing_args = np.array(list(zip(qualifiers,line))).flatten()
    parsing_args = self._clean_parsing_args(parsing_args)
    
    print(parsing_args)
    parser = argparse.ArgumentParser(description='Parses the data file input.')
    
    parser.add_argument('--sample-name', type=str, default='no-name-given')
    parser.add_argument('--setup-name', type=str, default='no-name-given')
    parser.add_argument('--xdet', type=float, default = -1)
    parser.add_argument('--xstart', type=int, default = -1)
    parser.add_argument('--xend', type=int, default = -1)
    parser.add_argument('--gamma-start', type=int, default = -1)
    parser.add_argument('--gamma-end', type=int, default = -1)
    parser.add_argument('--integration-type', type=str, default='each', choices=['each', 'mean'])
    
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
              '-xe', '{}'.format(args.xend),
              '-t', args.integration_type] + setup
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
