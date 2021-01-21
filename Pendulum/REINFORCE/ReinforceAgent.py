import numpy as np

def random_choice_prob_index(a, axis):
  r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
  return (a.cumsum(axis=axis) > r).argmax(axis=axis)

class ReinforceAgent:
  def __init__(self, model, actions, noise=0):
    self._model = model
    self._actions = actions
    self._noise = noise
    return
  
  def reset(self):
    return
  
  def process(self, state):
    return self.processBatch([state])[0]
    
  def processBatch(self, states):
    # actions = self._model.predict(np.array(states)) # SLOW
    actions = self._model(np.array(states)).numpy()

    indexes = random_choice_prob_index(actions, axis=1)
    if 0 < self._noise:
      rndIndexes = np.where(np.random.random_sample((indexes.shape[0], )) < self._noise)
      indexes[rndIndexes] = np.random.choice(
        np.arange(self._actions.N), indexes.shape
      )[rndIndexes]
    return self._actions.toValues(indexes)