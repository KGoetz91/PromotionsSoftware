#
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join, isfile

class P08Image:
  
  def __init__(self, path):
    try:
      isfile(path)
      self._IMAGE = np.array(Image.open(path,mode="r"))
    except:
      print('File {} has a problem.'.format(path))
    
  def show(self):
    plt.imshow(self._IMAGE)
    plt.show()
    plt.clf()
    
  def showlog(self):
    plt.imshow(np.log10(1+self._IMAGE))
    plt.show()
    plt.clf()
