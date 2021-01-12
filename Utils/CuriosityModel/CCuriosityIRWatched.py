import numpy as np

class CCuriosityIRWatched:
  def __init__(self, curiosity):
    self._curiosity = curiosity
    self.reset()
    return
  
  def reset(self):
    self._meanRewards = []
    self._maxReward = 0
    return
  
  def fit(self, *inputs, **kwargs):
    return self._curiosity.fit(*inputs, **kwargs)

  def rewards(self, *inputs):
    rewards = self._curiosity.rewards(*inputs)
    self._meanRewards.append(rewards.mean())
    self._maxReward = max((self._maxReward, rewards.max()))
    return rewards
  
  @property
  def info(self):
    return {
      'mean reward': np.array(self._meanRewards).mean(),
      'max reward': self._maxReward
    }