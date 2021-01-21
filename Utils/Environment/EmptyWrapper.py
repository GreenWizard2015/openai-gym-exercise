class EmptyWrapper:
  def __init__(self, env):
    self._env = env
    return
  
  def reset(self):
    return self._env.reset()
    
  def apply(self, *args):
    state, reward, done, prevState = self._env.apply(*args)
    return state, reward, done, prevState
  
  def render(self, *args, **kwargs):
    return self._env.render(*args, **kwargs)
  
  def hide(self):
    return self._env.hide()
  
  @property
  def state(self):
    return self._env.state
  
  @property
  def done(self):
    return self._env.done
  
  @property
  def score(self):
    return self._env.score
  