import numpy as np
import Utils.Environment as ENV
import gym

class BasicPendulumEnvironment(ENV.EmptyWrapper):
  def __init__(self):
    env = gym.make("Pendulum-v0")
    self._forceScale = env.action_space.high
    self._maxSpeed = env.max_speed
    
    super().__init__(
      ENV.ProcessedStateEnv(
        ENV.GymEnvironment(env),
        processor=self._processState
      )
    )
    return

  def apply(self, force):
    state, reward, done, prevState = self._env.apply( (self._forceScale * force,) )
    return state, reward[0], done, prevState

  def _processState(self, state):
    costh, sinth, speed = state
    return np.array((costh, sinth, speed / self._maxSpeed)).reshape((-1, ))

  @property
  def score(self):
    return self._env.score[0]