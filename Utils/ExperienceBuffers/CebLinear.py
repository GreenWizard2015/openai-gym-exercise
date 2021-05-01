import random
import numpy as np
import math
import itertools
from _collections import defaultdict

_WEIGHTS_MODES = {
  'abs': math.fabs,
  'reward': lambda x: x,
  'same': None
}

class CebLinear:
  def __init__(self, maxSize, sampleWeight='same'):
    self.maxSize = maxSize
    self._sizeLimit = math.floor(maxSize * 1.1)
    self._samples = []
    self._sampleWeight = _WEIGHTS_MODES.get(sampleWeight, sampleWeight)
  
  def addEpisode(self, replay, terminated):
    if 1 < len(replay):
      for step in replay[:-1]:
        self._samples.append((*step, 1))
    self._samples.append((*replay[-1], -1 if terminated else 0))

    self.update()
    return

  def update(self):
    if self._sizeLimit < len(self._samples):
      self._samples = self._samples[-self.maxSize:]
    return 
    
  def __len__(self):
    return len(self._samples)
  
  def _fixRewardMultiplier(self, x):
    if np.isscalar(x):
      return abs(x)

    if isinstance(x, (np.ndarray, np.generic)):
      return np.abs(x)
    
    raise Exception('Unknown reward type. (%s)' % type(x))
  
  def _createBatch(self, batch_size, sampler):
    samplesLeft = batch_size
    cumweights = None
    if self._sampleWeight:
      cumweights = list(itertools.accumulate(self._sampleWeight(x[2]) for x in self._samples))
      
    indexRange = np.arange(len(self._samples)) 
    res = defaultdict(list)
    while 0 < samplesLeft:
      indexes = set(random.choices(
        indexRange, cum_weights=cumweights,
        k=min((samplesLeft, len(self._samples)))
      ))

      for sInd in indexes:
        sample = sampler(sInd)
        if sample:
          for i, value in enumerate(sample[:-1]):
            res[i].append(value)
          res[len(sample) - 1].append(self._fixRewardMultiplier(sample[-1]))
          samplesLeft -= 1
    
    return [np.array(values) for values in res.values()]
    
  def sampleBatch(self, batch_size, **kwargs):
    return self._createBatch(batch_size, lambda i: self._samples[i])
  
  def sampleSequenceBatch(self, batch_size, sequenceLen, **kwargs):
    def sampler(ind):
      sample = self._samples[ind:ind+sequenceLen]
      if not (sequenceLen == len(sample)): return None
      if 1 < sequenceLen:
        if any(x[-1] < 1 for x in sample[:-1]):
          return None
      
      transposed = [
        np.array([x[col] for x in sample]) for col in range(len(sample[0]))
      ]
      return transposed
          
    return self._createBatch(batch_size, sampler)