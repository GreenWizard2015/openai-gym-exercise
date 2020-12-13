from Utils.Environment.EmptyWrapper import EmptyWrapper

class ProcessedStateEnv(EmptyWrapper):
  def __init__(self, env, processor):
    super().__init__(env)
    self._processor = processor
    
  def reset(self):
    return self._processor(self._env.reset())
    
  def apply(self, *args):
    state, reward, done, prevState = self._env.apply(*args)
    return self._processor(state), reward, done, self._processor(prevState)

  @property
  def state(self):
    return self._processor(self._env.state)