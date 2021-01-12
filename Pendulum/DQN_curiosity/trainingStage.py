import numpy as np

def train(model, memory, params):
  BOOTSTRAPPED_STEPS = params['steps']
  GAMMA = params['gamma']
  lossSum = 0
  for _ in range(params['episodes']):
    states, sampleActions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleSequenceBatch(
      batch_size=params['batchSize'],
      maxSamplesFromEpisode=params.get('maxSamplesFromEpisode', 16),
      sequenceLen=BOOTSTRAPPED_STEPS
    )
    actions = params['actions'].toIndex(sampleActions[:, 0])

    futureScores = model.predict(nextStates[:, -1]).max(axis=-1)
    futureScores *= nextStateScoreMultiplier[:, -1] * (GAMMA ** BOOTSTRAPPED_STEPS)
    
    totalRewards = (rewards * (GAMMA ** np.arange(BOOTSTRAPPED_STEPS))).sum(axis=-1)
    targets = model.predict(states[:, 0])
    allRows = np.arange(len(targets))
    targets[allRows, actions] = totalRewards + futureScores

    lossSum += model.fit(states[:, 0], targets, epochs=1, verbose=0).history['loss'][0]
    ###
    
  return lossSum / params['episodes']

def trainCuriosity(model, memory, params):
  lossSum = 0
  for _ in range(params['episodes']):
    states, sampleActions, *_ = memory.sampleBatch(
      batch_size=params['batchSize'],
      maxSamplesFromEpisode=params.get('maxSamplesFromEpisode', 16)
    )
    
    lossSum += model.fit(states, sampleActions).history['loss'][0]
    ###
    
  return lossSum / params['episodes']