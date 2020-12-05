import random
import numpy as np
import collections

class CebFromBuckets:
  def __init__(self, buckets, samplesPerBucketLimit):
    self._getBuckets = buckets if callable(buckets) else lambda: buckets
    self._samplesPerBucketLimit = samplesPerBucketLimit

  def addEpisode(self, replay, terminated):
    raise Exception('Unsupported operation.')

  def _createBatch(self, batch_size, sampler):
    _buckets = list(self._getBuckets())
    results = collections.defaultdict(list)
    while 0 < batch_size:
      samples = sampler(random.choice(_buckets), min((self._samplesPerBucketLimit, batch_size)))
      if samples:
        batch_size -= len(samples[0])
        for i, values in enumerate(samples):
          results[i].append(values)
      ###
    return [np.concatenate(tuple(values)) for values in results.values()]

  def sampleBatch(self, batch_size, *args, **kwargs):
    def sampler(bucket, N):
      return bucket.sampleBatch(N, *args, **kwargs)
     
    return self._createBatch(batch_size, sampler)

  def sampleSequenceBatch(self, batch_size, *args, **kwargs):
    def sampler(bucket, N):
      return bucket.sampleSequenceBatch(N, *args, **kwargs)
     
    return self._createBatch(batch_size, sampler)