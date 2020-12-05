import collections
from Utils.ExperienceBuffers.CebFromBuckets import CebFromBuckets

class CebHashed:
  def __init__(self, hasher, createBucket, bucketsCount, samplesPerBucketLimit):
    self._hasher = lambda x: abs(int(hasher(x))) % bucketsCount
    self._buckets = collections.defaultdict(createBucket)
    self._memory = CebFromBuckets(
      buckets=lambda: list(self._buckets.values()),
      samplesPerBucketLimit=samplesPerBucketLimit
    )

  def addEpisode(self, replay, terminated):
    bucketID = self._hasher(replay)
    return self._buckets[bucketID].addEpisode(replay, terminated)

  def sampleBatch(self, *args, **kwargs):
    return self._memory.sampleBatch(*args, **kwargs)

  def sampleSequenceBatch(self, batchSize, *args, **kwargs):
    return self._memory.sampleSequenceBatch(*args, **kwargs)