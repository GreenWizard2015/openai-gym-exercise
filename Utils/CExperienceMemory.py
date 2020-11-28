import random
import numpy as np
import math
import itertools

class CExperienceMemory:
  def __init__(self, maxSize):
    self.maxSize = maxSize
    self.sizeLimit = math.floor(maxSize * 1.1)
    self.episodes = []
    self.minScore = -math.inf
  
  def addEpisode(self, replay, terminated):
    score = sum(x[2] for x in replay) # state, action, 2 - reward
    if score < self.minScore: return
    self.episodes.append((replay, score, terminated))

    if self.sizeLimit < len(self.episodes):
      self.update()
    return

  def update(self):
    self.episodes = list(
      sorted(self.episodes, key=lambda x: x[1], reverse=True)
    )[:self.maxSize]
    self.minScore = self.episodes[-1][1]
    return 
    
  def __len__(self):
    return len(self.episodes)
  
  def sampleBatch(self, batch_size, maxSamplesFromEpisode=5):
    batchSize = 0
    cumweights = list(itertools.accumulate(x[1] for x in self.episodes))
    res = []
    while batchSize < batch_size:
      episode, _, wasTerminated = random.choices(self.episodes, cum_weights=cumweights, k=1)[0]
      lastActionScore = 1 if wasTerminated else 0
      
      minibatchIndexes = set(random.choices(
        np.arange(len(episode)),
        weights=[1 for x in episode],
        k=min((maxSamplesFromEpisode, batch_size - batchSize, len(episode)))
      ))
      
      for ind in minibatchIndexes:
        data = episode[ind]
        # populate res arrays
        while len(res) <= len(data):
          res.append([])

        for i, value in enumerate(data):
          res[i].append(value)
        res[-1].append(lastActionScore if ind == len(episode) - 1 else 1) # last action in replay?
        batchSize += 1

    return [np.array(values) for values in res]