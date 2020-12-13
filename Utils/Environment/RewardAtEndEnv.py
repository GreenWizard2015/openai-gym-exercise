from Utils.Environment.EmptyWrapper import EmptyWrapper
import numpy

class RewardAtEndEnv(EmptyWrapper):
  def __init__(self, env, use='reward'):
    super().__init__(env)
    self._use = use
    return
  
  def apply(self, *args):
    state, reward, done, prevState = self._env.apply(*args)
    if not done:
      return state, numpy.zeros_like(reward), done, prevState

    if 'reward' == self._use:
      return state, reward, done, prevState
    if 'score' == self._use:
      return state, numpy.zeros_like(reward) + self._env.score, done, prevState
    
    raise Exception('Unknown "use" value.')