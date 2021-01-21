import Utils.Environment as ENV
from Pendulum.BasicPendulumEnvironment import BasicPendulumEnvironment

class PendulumEnvironment(ENV.EmptyWrapper):
  def __init__(self, fixReward=False, seed=None):
    super().__init__(
      ENV.DeltaRewardEnv(
        BasicPendulumEnvironment(fixReward=fixReward, seed=seed)
      )
    )
    return