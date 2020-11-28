import random
import numpy as np

class DQNAgent:
  def __init__(self, model, exploreRate=0):
    self._model = model
    self._exploreRate = exploreRate
    return
  
  def reset(self):
    return
  
  def process(self, state):
    return self.processBatch([state])[0]
    
  def processBatch(self, states):
    actions = self._model.predict(np.array(states)).argmax(axis=-1)
    for index in range(len(actions)):
      if random.random() < self._exploreRate:
        actions[index] = random.random() - .5

    return np.where(0 < actions, 1, -1)