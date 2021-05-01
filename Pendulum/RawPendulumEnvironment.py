import Utils.Environment as ENV
import gym

class RawPendulumEnvironment(ENV.EmptyWrapper):
  def __init__(self, seed=None):
    env = gym.make("Pendulum-v0")
    self._forceScale = env.action_space.high
    super().__init__(
      ENV.ProcessedStateEnv(
        ENV.GymEnvironment(env, seed=seed),
        processor=self._processState
      )
    )
    return

  def apply(self, force):
    state, reward, done, prevState = self._env.apply( (self._forceScale * force,) )
    return state, reward, done, prevState

  @property
  def score(self):
    return self._env.score[0]
  
  def _processState(self, state):
    return state.reshape((-1, ))