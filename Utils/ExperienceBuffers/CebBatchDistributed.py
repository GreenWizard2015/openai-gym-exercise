import numpy
from Utils.ExperienceBuffers.CebFromBuckets import CebFromBuckets

class CebBatchDistributed:
  def __init__(self, buckets, samplesPerBucketLimit):
    self._buckets = buckets
    self._memory = CebFromBuckets(
      buckets=lambda: self._buckets,
      samplesPerBucketLimit=samplesPerBucketLimit
    )
  
  def addEpisode(self, replay, terminated):
    raise Exception('Use addEpisodes.')
  
  def sampleBatch(self, *args, **kwargs):
    return self._memory.sampleBatch(*args, **kwargs)
  
  def sampleSequenceBatch(self, *args, **kwargs):
    return self._memory.sampleSequenceBatch(*args, **kwargs)
  
  def addEpisodes(self, replays):
    scores = [sum((x[2] for x in replay)) for replay, _ in replays]
    replaysByScore = list(sorted(numpy.arange(len(replays)), key=lambda x: scores[x], reverse=True))
    for i, subrangeIndexes in enumerate(numpy.array_split(replaysByScore, len(self._buckets))):
      for j in subrangeIndexes:
        self._buckets[i].addEpisode(*replays[j])
    return