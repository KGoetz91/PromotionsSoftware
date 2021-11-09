##############################################################
#Worker module, parsing the input parameters and choosing    #
#the correct calibrator for the data.                        #
#Author: Klaus Götz                                          #
#E-Mail: kgoetz91@t-online.de                                #
#License: GPL-3.0                                            #
#09. November 2021                                           #
##############################################################

from .workload import Workload 
import argparse

class Worker:
  """Worker class that parses imput arguments, handles the creation of correct workload and calibrator classes and start the reduction process."""
  def __init__(self, params):
    """The starting parameters are initialized and a workload is created.
    
    Argument params should be the starting parameters, e.g. sys.argv[1:].
    """
    self._init_parameters(params)
    self.workload = Workload(self._SETUP, self._DATA)
    
    if self._VERBOSE:
      print(self.workload.setups())
      print(self.workload.workload())
  
  def _init_parameters(self, params):
    """Parameter initialization.
    
    
    """
    self._ALLOWED_TYPES = ['saxans']
    parser = argparse.ArgumentParser()
    
    parser.add_argument('setup', type=str, help='File containing the setups that will be loaded.')
    parser.add_argument('data', type=str, help='File containing the formatted data information.')
    parser.add_argument('-v', '--verbose', type=bool, help='Display steps along the way.', default=False, const=True, nargs='?')
    parser.add_argument('-t', '--type', type=str,
                        help='Which SAXS data is to be refined.', 
                        default = 'saxans',
                        choices=self._ALLOWED_TYPES)
    parser.add_argument('-o', '--output', type=str, default = './',
                        help='Path were the outputs will be saved.')
    
    args = parser.parse_args(params)
    self._SETUP = args.setup
    self._DATA = args.data
    self._VERBOSE = args.verbose
    self._TYPE = args.type
    self._OUTP = args.output
    
    if self._TYPE == "saxans":
      from .calibrator import SAXANSCalibrator as Calibrator 
      
    self._CALIBRATOR = Calibrator
  
  def work(self):
    """Create a calibrator for each dataset and do the calibration."""
    for data in self.workload.workload().keys():
      work = self.workload.workload()[data]+['-v',str(self._VERBOSE), '-o', self._OUTP]
      integrator = self._CALIBRATOR(data, work)
      result = integrator.integrate()
