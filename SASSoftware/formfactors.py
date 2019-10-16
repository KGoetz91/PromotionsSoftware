#!/usr/bin/python3
from scipy.integrate import dblquad
import numpy as np
from math import sin, cos
from math import pi as PI
import asyncio

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
  
  def intensity(self):
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
 
  @asyncio.coroutine
  def intent(self, q, loop):
    print('Started calculation q: {}'.format(q))
    if self.__FORM__.isotropic():
      res = (q, (self.__FORM__.form_factor(q)*
              self.__FORM__.form_factor(q)))
      print('Finished calculation q: {} int: {}'.format(q, res[1]))
      return res
    else: 
      ff = lambda theta, phi: self.__FORM__.form_factor(q, theta, phi)
      integrand = lambda theta, phi: ff(theta, phi)*ff(theta,phi)
      res = yield from loop.run_in_executor(None, dblquad,
                        integrand, 0, PI/2, 0, PI/2)
      res = res[0]
      print('Finished calculation q: {} int: {}'.format(q, res))
      return (q, 8*res)
      
 
  def intensity(self, q_range):
    tasks = []
    loop = asyncio.get_event_loop()
    for q in q_range:
      tasks.append(asyncio.async(self.intent(q, loop)))
    result = loop.run_until_complete(asyncio.gather(*tasks))
    return result
      
if __name__ == '__main__':
  print('There will be help.')
