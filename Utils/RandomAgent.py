import random
import numpy

class RandomAgent:
  def __init__(self, low, high):
    self._low = low
    self._high = high
    return
  
  def reset(self):
    pass
  
  def _randomAction(self):
    return self._low + random.random() * (self._high - self._low)
  
  def process(self, state):
    return self._randomAction()
  
  def processBatch(self, states):
    return numpy.array([ self._randomAction() for _ in states ])