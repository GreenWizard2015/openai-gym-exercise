from Utils.ExperienceBuffers.CebHashed import CebHashed
import random
import sys

class CebRandomlyDistributed:
  def __init__(self, *args, **kwargs):
    self._memory = CebHashed(
      *args, **kwargs,
      hasher=lambda: random.randint(0, sys.maxsize)
    )
  
  def addEpisode(self, replay, terminated):
    return self._memory.addEpisode(replay, terminated)

  def sampleBatch(self, *args, **kwargs):
    return self._memory.sampleBatch(*args, **kwargs)
  
  def sampleSequenceBatch(self, *args, **kwargs):
    return self._memory.sampleSequenceBatch(*args, **kwargs)