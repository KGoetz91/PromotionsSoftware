#!/usr/bin/python3

import os,ctypes
from scipy import integrate, LowLevelCallable

lib = ctypes.CDLL(os.path.abspath('ff_lib.so'))
lib.ff_cube.restype = ctypes.c_double
lib.ff_cube.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

q_range = range(0,100)
for q in q_range:
  
  FiveDouble = ctypes.c_double *5
  c = FiveDouble(1, 1, 0, 5,q)
  
  user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
  func = LowLevelCallable(lib.ff_cube, user_data)
  integrate.nquad(func, [[0,1.55],[0,1.55]])
