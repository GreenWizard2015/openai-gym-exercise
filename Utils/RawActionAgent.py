import numpy as np

class RawActionAgent:
  def __init__(self, model, processor=None):
    self._model = model
    self._processor = processor if processor else lambda x: x
    return
  
  def reset(self):
    return
  
  def process(self, state):
    return self.processBatch([state])[0]
    
  def processBatch(self, states):
    actions = self._model(np.array(states)).numpy()
    return self._processor(actions)