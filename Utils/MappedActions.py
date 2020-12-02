import numpy as np

class MappedActions:
  def __init__(self, N, valuesRange=[0., 1.]):
    self._indexes = np.linspace(valuesRange[0], valuesRange[1], N + 1)[1:-1]
    self._values = np.linspace(valuesRange[0], valuesRange[1], N)
    self.N = N
  
  def toIndex(self, values):
    return np.digitize(values, self._indexes)
  
  def toValues(self, indexes):
    return self._values[indexes]