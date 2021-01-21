import gym

class GymEnvironment:
  def __init__(self, env, seed=None):
    self._env = gym.make(env) if isinstance(env, str) else env
    self._seed = seed
    return
  
  def reset(self):
    self._env.seed(self._seed)
    self._state = self._env.reset()
    self._done = False
    self._score = 0
    return self._state
    
  def apply(self, *args):
    prevState = self.state
    state, reward, done, _ = self._env.step(*args)
    self._state = state
    self._done = done
    self._score += 0 if done else reward
    return state, reward, done, prevState
  
  def render(self, mode='human', *args, **kwargs):
    return self._env.render(*args, mode=mode, **kwargs)
  
  def hide(self):
    return self._env.close()
  
  @property
  def state(self):
    return self._state
  
  @property
  def done(self):
    return self._done
  
  @property
  def score(self):
    return self._score
  