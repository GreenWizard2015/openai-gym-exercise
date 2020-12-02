import numpy as np

class DQNAgent:
  def __init__(self, model, actions, exploreRate=0):
    self._model = model
    self._exploreRate = exploreRate
    self._actions = actions
    return
  
  def reset(self):
    return
  
  def process(self, state):
    return self.processBatch([state])[0]
    
  def processBatch(self, states):
    actions = self._model.predict(np.array(states))
    if 0 < self._exploreRate:
      rndIndexes = np.where(np.random.random_sample((actions.shape[0], )) < self._exploreRate)
      actions[rndIndexes] = np.random.random_sample(actions.shape)[rndIndexes]

    return self._actions.toValues(actions.argmax(axis=-1))