#!/usr/bin/python3
from scipy.integrate import dblquad
import numpy as np
from math import sin, cos
from math import pi as PI
import asyncio

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
    ff = lambda theta, phi : self.form_factor(q, theta, phi)
    integrand = lambda theta, phi: ff(theta, phi) * ff(theta, phi)
    res = yield from self.__LOOP__.run_in_executor(None, dblquad, 
                          integrand, 0, PI/2, 0, PI/2)
    res = 8*res[0]
    print('q: {}, int: {}'.format(q, res))
    return (q, res)
  
  #async def count_tasks(self, tasks):
    #while True:
      #asyncio.sleep(1)
      #i = 0
      #for task in tasks:
        #if not task.done():
          #i += 1
      #if i == 0:
        #break
      #print('{} calculations still running.'.format(i))
    #return 0
    
  def scatter(self, q_range):
    tasks = []
    for q in q_range:
      tasks.append(asyncio.async(self.intensity(q)))
    result = self.__LOOP__.run_until_complete(
                                        asyncio.gather(*tasks))
    return result
    
    
class CubeOneShell(Cube):
  
  def __init__(self, epsilon, eta_cube, eta_shell, eta_solv,
               edge_cube, t_shell):
    super().__init__(epsilon, eta_cube, eta_solv,
                     edge_cube+t_shell)
    self.core = Cube(epsilon, eta_cube, eta_shell, edge_cube)
    self.shell = Cube(epsilon, eta_shell, eta_solv,
                      edge_cube+t_shell)
  
  def form_factor(self, q, theta, phi):
    return (self.core.form_factor(q, theta, phi)
            + self.shell.form_factor(q, theta, phi))

if __name__ == '__main__':
  print('There will be help.')
