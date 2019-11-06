#!/usr/bin/python3
from scipy.integrate import nquad
from scipy import LowLevelCallable
import numpy as np
from math import sin, cos
from math import pi as PI
import asyncio
import os, ctypes

class FormFactors:
  
  def __init__(self, epsilon = None, volume = None):
    if epsilon == None:
      self.__EPSILON__ = 0
    else:
      self.__EPSILON__ = float(epsilon)
      
    if volume == None:
      self.__VOLUME__ = 0
    else:
      self.__VOLUME__ = volume
    
  def sinc(self, x):
    return sin(x)/x
  
  def volume(self):
    return self.__VOLUME__
  
  def form_factor(self):
    pass
  
  def intensity(self, q):
    pass
  
  def scatter(self, q_range):
    pass

class Cube(FormFactors):
  
  def __init__(self, epsilon, eta_in, eta_out, edge):
    super().__init__(epsilon, edge**3)
    self.__ETA_IN__ = eta_in
    self.__ETA_OUT__ = eta_out
    self.__EDGE__ = edge
    self.__LOOP__ = asyncio.get_event_loop()
    
    lib = ctypes.CDLL(os.path.abspath('ff_lib.so'))
    self.__FF__ = lib.int_cube
    self.__FF__.restype = (ctypes.c_double)
    self.__FF__.argtypes = (ctypes.c_int,
                ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
    self.ParameterType = ctypes.c_double * 5
    
    
  #Deprecated since the use of c library
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
    
  @asyncio.coroutine
  def intensity(self, q):
    print('Calculating intensity for q: {}'.format(q))
    
    params = self.ParameterType(self.__EPSILON__,
                  self.__ETA_IN__, self.__ETA_OUT__,
                  self.__EDGE__, q)
    params = ctypes.cast(ctypes.pointer(params), ctypes.c_void_p)
    integrand = LowLevelCallable(self.__FF__, params)
    res = nquad(integrand, [[0, PI/2],[0, PI/2]])
    res = 8*res[0]
    print('q: {}, int: {}'.format(q, res))
    return (q, res)
  
  def scatter(self, q_range):
    tasks = []
    for q in q_range:
      tasks.append(asyncio.async(self.intensity(q)))
    result = self.__LOOP__.run_until_complete(
                                        asyncio.gather(*tasks))
    return result
  
class CubeOneShell(FormFactors):
  
  def __init__(self, epsilon, eta_cube, eta_shell, eta_solv,
               edge_cube, t_shell):
    super().__init__(epsilon,(edge_cube+t_shell)**3)
    self.ParameterType = ctypes.c_double * 7
    self.__PARAMS__ = self.ParameterType(epsilon, eta_cube, eta_shell, eta_solv, edge_cube, t_shell, 0)
    self.__LOOP__ = asyncio.get_event_loop()
    
    lib = ctypes.CDLL(os.path.abspath('ff_lib.so'))
    self.__FF__ = lib.int_cube_one_shell
    self.__FF__.restype = (ctypes.c_double)
    self.__FF__.argtypes = (ctypes.c_int,
                ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)
    
  def scatter(self, q_range):
    tasks = []
    for q in q_range:
      tasks.append(asyncio.async(self.intensity(q)))
    result = self.__LOOP__.run_until_complete(
                                        asyncio.gather(*tasks))
    return result
  
  @asyncio.coroutine
  def intensity(self, q):
    print('Calculating intensity for q: {}'.format(q))
    
    params = self.__PARAMS__
    params[6] = q
    params = ctypes.cast(ctypes.pointer(params), ctypes.c_void_p)
    integrand = LowLevelCallable(self.__FF__, params)
    res = nquad(integrand, [[0, PI/2],[0, PI/2]])
    res = 8*res[0]
    print('q: {}, int: {}'.format(q, res))
    return (q, res)
    

if __name__ == '__main__':
  print('There will be help.')
