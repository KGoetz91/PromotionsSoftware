##############################################################
#A set of helper functions for the sasreducer package.       #
#Example: gaussian function, etc.                            #
#Author: Klaus Götz                                          #
#E-Mail: kgoetz91@t-online.de                                #
#License: GPL-3.0                                            #
#09. November 2021                                           #
##############################################################

import numpy as np

def gaussian(x,x0,sigma,maximum):
  """Calculates a gaussian distribution.  
  
  Calculates the gaussion distribution at the values of an array x.
  x0: center of gaussian
  sigma: width of gaussian
  maximum: amplitude of gaussian
  """
  dx = np.subtract(x,x0)
  quotient = -2*sigma**2
  exponent = np.divide(np.power(dx,2),quotient)
  return maximum*np.exp(exponent)
