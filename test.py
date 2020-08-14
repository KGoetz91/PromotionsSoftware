#!/usr/bin/python3

from Generic2DSASReducer.loader.saxsans import SaxansLoader
import matplotlib.pyplot as plt

import numpy as np
import argparse

if __name__ == '__main__':
  
  a = SaxansLoader()
  data_files = list(range(27551,27555+1,1))
  gamma_files = list(range(27556,27560+1,1))
  
  #result = a.load_set('/mnt/d/ILLAug2020_LTP9-9/rawdata', files)
  