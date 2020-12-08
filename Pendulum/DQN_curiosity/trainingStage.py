import tensorflow
import numpy as np

def train(model, memory, curiousityModel, params):
  modelClone = tensorflow.keras.models.clone_model(model)
  modelClone.set_weights(model.get_weights()) # use clone model for stability
  
  BOOTSTRAPPED_STEPS = params['steps']
  GAMMA = params['gamma']
  curiousitySum = 0
  lossSum = 0
  for episode in range(params['episodes']):
    states, sampleActions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleSequenceBatch(
      batch_size=params['batchSize'],
      maxSamplesFromEpisode=params.get('maxSamplesFromEpisode', 16),
      sequenceLen=BOOTSTRAPPED_STEPS
    )
    actions = params['actions'].toIndex(sampleActions[:, 0])
    
    ###############
    curiousity = np.zeros_like(rewards)
    for step in range(params['curiousity range']):
      curiousityInput = np.concatenate((states[:, step], sampleActions[:, step].reshape((-1, 1))), axis=1)
      curiousityOutput = nextStates[:, step]
      predictedState = curiousityModel.predict(curiousityInput)
      curiousity[:, step] = np.sqrt(np.power(predictedState - curiousityOutput, 2).sum(axis=-1))
    
    curiousitySum += np.sum(curiousity.max(axis=-1)) / len(curiousity)
    ###############

    futureScores = modelClone.predict(nextStates[:, -1]).max(axis=-1)
    futureScores *= nextStateScoreMultiplier[:, -1] * (GAMMA ** BOOTSTRAPPED_STEPS)
    
    totalRewards = (rewards * (GAMMA ** np.arange(BOOTSTRAPPED_STEPS))).sum(axis=-1)
    targets = modelClone.predict(states[:, 0])
    allRows = np.arange(len(targets))
    targets[allRows, actions] = totalRewards + futureScores
    targets[allRows, actions] = params['curiousity function'](targets[allRows, actions], curiousity)

    lossSum += model.fit(states[:, 0], targets, epochs=1, verbose=0).history['loss'][0]
    if (episode % params.get('curiousity update rate', 1)) == 0:
      curiousityModel.fit(
        np.concatenate((states[:, 0], sampleActions[:, 0].reshape((-1, 1))), axis=1),
        nextStates[:, 0],
        epochs=1, verbose=0
      )
    ###
    
  print('Avg. curiousity: %.4f' % (curiousitySum / params['episodes']))
  return lossSum / params['episodes']