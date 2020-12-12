import tensorflow
import numpy as np

def train(model, memory, params):
  modelClone = tensorflow.keras.models.clone_model(model)
  modelClone.set_weights(model.get_weights()) # use clone model for stability
  
  BOOTSTRAPPED_STEPS = params['steps']
  GAMMA = params['gamma']
  lossSum = 0
  for _ in range(params['episodes']):
    states, actions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleSequenceBatch(
      batch_size=params['batchSize'],
      maxSamplesFromEpisode=params.get('maxSamplesFromEpisode', 16),
      sequenceLen=BOOTSTRAPPED_STEPS
    )
    actions = params['actions'].toIndex(actions[:, 0])

    futureScores = modelClone.predict(nextStates[:, -1]).max(axis=-1) * nextStateScoreMultiplier[:, -1]
    totalRewards = (rewards * (GAMMA ** np.arange(BOOTSTRAPPED_STEPS))).sum(axis=-1)
    targets = modelClone.predict(states[:, 0])
    targets[np.arange(len(targets)), actions] = totalRewards + futureScores * (GAMMA ** BOOTSTRAPPED_STEPS)

    lossSum += model.fit(states[:, 0], targets, epochs=1, verbose=0).history['loss'][0]
    ###
  return lossSum / params['episodes']