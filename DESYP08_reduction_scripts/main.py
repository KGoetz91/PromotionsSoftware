#!/usr/bin/python3

import sys
from os.path import join, isfile
from os import listdir
import numpy as np

from p08reduction.imageloader import P08Image

if __name__ == '__main__':
  
  path = '/home/klaus/Documents/DESYSeptember21Vorbereitung/Daten/raw'
  filename = 'W9_emptyCell_'
  first_number = 255
  last_number = 261
  
  numbers = np.array(range(first_number, last_number+1))
  
  for i in numbers:
    full_path = join(path, "{}{:05d}/p100k/".format(filename,i))
    scan_files = list(listdir(full_path))
    print(scan_files)
    for scanpoint in scan_files:
      image = P08Image(join(full_path,scanpoint))
      image.showlog()
  
