from Utils.Environment.EmptyWrapper import EmptyWrapper
import numpy as np

class DeltaRewardEnv(EmptyWrapper):
  def reset(self):
    self._reward = None
    return self._env.reset()
    
  def apply(self, *args):
    state, reward, done, prevState = self._env.apply(*args)
    if np.isscalar(reward):
      rewardDelta = reward if self._reward is None else reward - self._reward
    else:
      rewardDelta = np.zeros_like(reward) if self._reward is None else reward - self._reward

    self._reward = reward
    return state, rewardDelta, done, prevState
