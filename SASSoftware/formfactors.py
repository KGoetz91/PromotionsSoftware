#!/usr/bin/python3
from scipy.integrate import dblquad
import numpy as np
from math import sin, cos
from math import pi as PI
from multiprocessing.pool import ThreadPool

class FormFactors:
  
  def __init__(self, epsilon = None, volume = None, is_isotropic = False):
    if epsilon == None:
      self.__EPSILON__ = 0
    else:
      self.__EPSILON__ = float(epsilon)
      
    if volume == None:
      self.__VOLUME__ = 0
    else:
      self.__VOLUME__ = volume
    
    if is_isotropic:
      self.__ISO__ = True
    else:
      self.__ISO__ = False
  
  def sinc(self, x):
    return sin(x)/x
  
  def volume(self):
    return self.__VOLUME__
  
  def isotropic(self):
    return self.__ISO__
  
  def form_factor(self):
    pass
  
class Cube(FormFactors):
  
  def __init__(self, epsilon, eta_in, eta_out, edge):
    super().__init__(epsilon, edge**3)
    self.__ETA_IN__ = eta_in
    self.__ETA_OUT__ = eta_out
    self.__EDGE__ = edge
    
  def form_factor(self, q, theta, phi):
    delta_rho = self.__ETA_IN__-self.__ETA_OUT__
    v_cube = self.__VOLUME__
    if q < self.__EPSILON__:
      return (delta_rho*v_cube)
    else:
      r = self.__EDGE__/2
      qx = -q*sin(theta)*cos(phi)
      qy = q*sin(theta)*sin(phi)
      qz = q*cos(theta)
      
      return (delta_rho*v_cube*self.sinc(qx*r)
              *self.sinc(qy*r)*self.sinc(qz*r))
    
class CubeOneShell(FormFactors):
  
  def __init__(self, epsilon, eta_cube, eta_shell, eta_solv,
               edge_cube, t_shell):
    volume = (edge_cube+t_shell)**3
    super().__init__(epsilon, volume)
    self.core = Cube(epsilon, eta_cube, eta_shell, edge_cube)
    self.shell = Cube(epsilon, eta_shell, eta_solv,
                      edge_cube+t_shell)
  
  def form_factor(self, q, theta, phi):
    return (self.core.form_factor(q, theta, phi)
            + self.shell.form_factor(q, theta, phi))

class Intensities:
  
  def __init__(self, formfactor):
    if isinstance(formfactor, FormFactors):
      self.__FORM__ = formfactor
    else:
      raise TypeError('Wrong formfactor input.')
  
  def intensity(self, q_range):
    i = np.array([])
    pool = ThreadPool()
    for q in q_range:
      print(q)
      if self.__FORM__.isotropic():
        i = np.append(i, self.__FORM__.form_factor(q))
      else:
        def integrand(theta, phi):
          return(self.__FORM__.form_factor(q, theta, phi)*
                  self.__FORM__.form_factor(q, theta, phi))
        new_i = pool.apply_async(dblquad, (integrand, 0,2*PI, 0, PI))
        #new_i = dblquad(integrand, 0, 2*PI, 0, PI)[0]/(4*PI)
        i = np.append(i, new_i.get()[0]/(4*PI))
    return i
      
if __name__ == '__main__':
  print('There will be help.')
