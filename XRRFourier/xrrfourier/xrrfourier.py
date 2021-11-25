
from os.path import isfile
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import blackman

import matplotlib.pyplot as plt

def create_sinus():
  N = 3600
  spacing = 2./N
  x = np.linspace(0.0, N*spacing, N, endpoint= False)
  
  freq1 = 50
  freq2 = 500
  
  y = np.sin(freq1 * 2.0 * np.pi * x) + 0.5*np.sin(freq2 * 2.0 * np.pi * x)
  
  return (np.array(x), np.array(y))

class XRRFourier:

  def lin_interpolate(self,x1,y1,e1,x2,y2,e2,x_det):
    a = (y2-y1)/(x2-x1)
    ea = np.sqrt((e1*e1)+(e2*e2))
    eb = np.sqrt((e1*e1)+(e2*e2)+(ea*ea))
    b = 0.5*(y2+y1-a*(x1+x2))
    ydet = a*x_det+b
    edet = np.sqrt((x_det*x_det*ea*ea)+(eb*eb))
    return x_det,ydet,edet

  def spline_data(self,x0, data):
    x1 = np.array(data[0])
    y1 = np.array(data[1])
    e1 = np.array(data[2])

    y2 = []
    e2 = []

    for x in x0:
      eq_ind = np.where(x1==x)
      if len(eq_ind[0]) == 1:
        y2.append(y1[eq_ind[0][0]])
        e2.append(e1[eq_ind[0][0]])
      else:
        min_ind = np.where(x1<x)[0]
        max_ind = np.where(x1>x)[0]
        if len(min_ind) == 0:
          xx1 = x1[max_ind[0]]
          xx2 = x1[max_ind[1]]
          yy1 = y1[max_ind[0]]
          yy2 = y1[max_ind[1]]
          ee1 = e1[max_ind[0]]
          ee2 = e1[max_ind[1]]
          xdet, ydet, edet = self.lin_interpolate(xx1,yy1,ee1,xx2,yy2,ee2,x)
        elif len(max_ind) == 0:
          xx1 = x1[min_ind[-1]]
          xx2 = x1[min_ind[-2]]
          yy1 = y1[min_ind[-1]]
          yy2 = y1[min_ind[-2]]
          ee1 = e1[min_ind[-1]]
          ee2 = e1[min_ind[-2]]
          xdet, ydet, edet = self.lin_interpolate(xx1,yy1,ee1,xx2,yy2,ee2,x)
        else:
          xx1 = x1[max_ind[0]]
          yy1 = y1[max_ind[0]]
          ee1 = e1[max_ind[0]]
          xx2 = x1[min_ind[-1]]
          yy2 = y1[min_ind[-1]]
          ee2 = e1[min_ind[-1]]
          xdet, ydet, edet = self.lin_interpolate(xx1,yy1,ee1,xx2,yy2,ee2,x)

        y2.append(ydet)
        e2.append(edet)
      
    return x0,y2,e2

  def load_XRRdata(self, filename):
    try:
      isfile(filename)
    except:
      print('Opening file {} resulted in a problem.'.format(filename))
      
    qs = []
    intensities = []
    errors = []
      
    with open(filename, 'r') as readfile:
      for line in readfile:
        if not line.startswith('#'):
          q, reflectivity, error = line.split('\t')
          qs.append(float(q))
          intensities.append(float(reflectivity))
          errors.append(float(error))
    
    return (np.array(qs), np.array(intensities), np.array(errors))

  def calculate_fourier(self, xrrdata):
    q = xrrdata[0]
    N = len(xrrdata[0])
    
    linear_qs = np.linspace(q[0],q[-1],N)
    linear_data = self.spline_data(linear_qs, xrrdata)
    sample_spacing = linear_data[0][1]-linear_data[0][0]
    
    normal_fourier = fft(linear_data[1])
    blackman_fourier = fft(np.array(linear_data[1])*blackman(N))
    frequencies = fftfreq(N, sample_spacing)[:N//2]
    
    plt.plot(frequencies[1:N//2], 2.0/N * np.abs(normal_fourier[1:N//2]), label="normal")
    plt.plot(frequencies[1:N//2], 2.0/N * np.abs(blackman_fourier[1:N//2]), label="blackman")
    plt.legend()
    plt.show()
    plt.clf()
    
    

  def __init__(self, filename):
    self._XRRDATA = self.load_XRRdata(filename)
    self._FOURIERDATA = self.calculate_fourier(self._XRRDATA)
  
  def test(self):
    data = create_sinus()
    plt.plot(data[0], data[1])
    plt.show()
    plt.clf()
    self.calculate_fourier((data[0], data[1], np.sqrt(data[1])))
