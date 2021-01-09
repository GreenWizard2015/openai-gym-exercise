from Utils.Environment.EmptyWrapper import EmptyWrapper

class DeltaRewardEnv(EmptyWrapper):
  def reset(self):
    self._reward = None
    return self._env.reset()
    
  def apply(self, *args):
    state, reward, done, prevState = self._env.apply(*args)
    
    if self._reward is None:
      rewardDelta = reward * 0.0 # preserve same type as reward
    else:
      rewardDelta = reward - self._reward

    self._reward = reward
    return state, rewardDelta, done, prevState
