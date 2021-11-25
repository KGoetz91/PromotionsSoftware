#!/usr/bin/python3

from xrrfourier.xrrfourier import XRRFourier as trafo

if __name__ == '__main__':
  
  filename = '/home/klaus/Documents/DESYSeptember21Vorbereitung/Daten/processed/reductions_211110/W9_emptyCell_cor_262.dat'
  fourier_data = trafo(filename)
  fourier_data.test()
