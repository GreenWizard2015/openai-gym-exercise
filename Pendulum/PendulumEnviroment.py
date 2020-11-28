import numpy as np
import gym

class PendulumEnviroment:
  def __init__(self):
    self._env = gym.make("Pendulum-v0")
    self.reset()
    return
  
  def reset(self):
    self.score = 0
    self._reward = None
    self.done = False
    return self._processState(self._env.reset())
  
  def apply(self, force):
    ''' force in range -1..1 '''
    prevState = self.state
    state, reward, done, _ = self._env.step(
      action=(self._env.action_space.high * force,)
    )
    reward = reward[0]
    self.score += 0 if self.done else reward
    # reward delta
    reward_ = (reward - self._reward) if self._reward is not None else 0
    self._reward = reward
    self.done = done
    return self._processState(state), reward_, done, prevState

  def _processState(self, state):
    costh, sinth, speed = state
    self._state = np.array((costh, sinth, speed / self._env.max_speed)).reshape((-1, ))
    return self._state
  
  def render(self):
    return self._env.render(mode='human')
  
  def hide(self):
    return self._env.close()
  
  @property
  def state(self):
    return self._state