#

import numpy as np

class Masks:
  def _saxans(self, data):
    """Creates a numpy array to be used as mask in pyFAI.
    
    The numpy array contains 0 for pixels that will be used and
    1 for pixels that are masked. The function automatically masks 
    all pixels with counts lower than 0 and the 3 dead pixels in
    the detector.
    """
    mask = np.array(data)
    mask[mask >= 0] = 0 
    mask[mask < 0] = 1 
    mask[666,388] = 1 
    mask[764,409] = 1 
    mask[1006,274] = 1 
    
    return mask

  def _saxspace(self, data):
    mask = np.array(self._twoddata)
    #y = 388
    #x = 398
    #print(mask[x,y])
    #print(mask[x-2:x+2,y-2:y+2])
    mask[mask >= 0] = 0
    mask[mask < 0] = 1
    mask2 = np.array(mask)
    
    radius = 30
    alpha = 45
    coordinates = []
    for x in range(-radius, radius, 1):
      for y in range(-radius, radius, 1):
        if x*x+y*y<=radius*radius:
          if (x>0) or (abs(y) > abs(x*np.tan(alpha*np.pi/180))):
            coordinates.append((x,y))
        
    coordinates = [(x+int(self._com[0])+1,y+int(self._com[1])) for (x,y) in coordinates]
    for (x,y) in coordinates:
      mask2[x,y] = 1000
      
    #plt.imshow(mask2)
    plt.imshow(np.log10(np.multiply(self._twoddata,np.multiply(-1,np.subtract(mask2,1)))))
    plt.show()
    
    #mask2[int(self._com[0])-10:int(self._com[0])+20, int(self._com[1])-10: int(self._com[1])+20] = 1
    #data_temp = np.multiply(self._twoddata, np.multiply(-1,np.subtract(mask2,1)))
    #mask[300,409] = 1 #mask[398,388] = 1 # Works only for my PC at home
    mask[666,388] = 1 
    mask[764,409] = 1 
    mask[1006,274] = 1 
    #mask[:int(self._com[0])]=1 # Works only for my PC at home
    mask[int(self._com[0])+1:]=1
    #np.save('mask.npy', mask)
    if self._VERBOSE:
      plt.imshow(np.log10(np.multiply(self._twoddata,np.multiply(-1,np.subtract(mask,1)))))
      plt.show()
    return mask
  
  def __init__(self, mask_type, data):
    self.supported_types = {'saxans': self._saxans,'saxspace': self._saxspace}
    if not(mask_type in self.supported_types.keys()):
      error_message = 'Supported mask types are:'
      for t in self.supported_types.keys:
        error_message += ' {}'.format(t)
      error_message += '.'
      raise ValueError(error_message)
    self.mask = self.supported_types[mask_type](data)
    
  def mask_outside_circle(self,com,radius):
    new_mask = self.mask
    x,y = self.mask.shape
    x_range = np.array(range(x))
    y_range = np.array(range(y))
    
    for x in x_range:
      for y in y_range:
        dx = x-com[0]
        dy = y-com[1]
        if dx*dx + dy*dy > radius*radius:
          new_mask[x][y]=1
          
    return new_mask
