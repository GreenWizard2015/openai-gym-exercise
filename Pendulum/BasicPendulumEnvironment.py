import numpy as np
import Utils.Environment as ENV
import gym
from math import sqrt

class BasicPendulumEnvironment(ENV.EmptyWrapper):
  def __init__(self, fixReward=False):
    env = gym.make("Pendulum-v0")
    self._forceScale = env.action_space.high
    self._maxSpeed = env.max_speed
    self._fixReward = fixReward
    
    super().__init__(
      ENV.ProcessedStateEnv(
        ENV.GymEnvironment(env),
        processor=self._processState
      )
    )
    return

  def apply(self, force):
    state, reward, done, prevState = self._env.apply( (self._forceScale * force,) )
    if self._fixReward:
      reward = self._fixedReward(state)
    else:
      reward = reward[0]
    return state, reward, done, prevState

  def _fixedReward(self, state):
    costh, sinth = state[:2]
    dist = (1. - costh) ** 2 + sinth ** 2
    return -sqrt(dist) if 0 < dist else 0.0
  
  def _processState(self, state):
    costh, sinth, speed = state
    return np.array((costh, sinth, speed / self._maxSpeed)).reshape((-1, ))

  @property
  def score(self):
    return self._env.score[0]