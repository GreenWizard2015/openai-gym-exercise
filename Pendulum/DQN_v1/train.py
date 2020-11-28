# -*- coding: utf-8 -*-
import tensorflow as tf
# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4 * 1024)]
)

import time
import numpy as np

from Pendulum.PendulumEnviroment import PendulumEnviroment
from Pendulum.DQN_v1.DQNAgent import DQNAgent

from Utils import emulate, emulateBatch, plotData2file
from Utils.CExperienceMemory import CExperienceMemory
from Utils.RandomAgent import RandomAgent

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def createModel(input_shape, output_size=2):
  inputs = res = layers.Input(shape=input_shape)
  res = layers.Dense(16, activation='relu')(res)
  res = layers.Dense(8, activation='relu')(res)
  res = layers.Dense(8, activation='relu')(res)
  res = layers.Dense(4, activation='relu')(res)
  return keras.Model(
    inputs=inputs,
    outputs=layers.Dense(output_size, activation='linear')(res)
  )

metrics = {
  'scores': {
    'avg.': [],
    'top 10%': [],
    'top 50%': [],
    'top 90%': [],
  }
}

env = PendulumEnviroment()
memory = CExperienceMemory(maxSize=5000)
# collect random experience
agent = RandomAgent(low=-1, high=1)
for episodeN in range(1000):
  replay, done = emulate(env, agent)
  memory.addEpisode(replay, terminated=not done)
print('random experience collected')
#######################################
# train agent
BATCH_SIZE = 512
TRAIN_EPISODES = 50
TEST_EPISODES = 256
EXPLORE_RATE = .5
EXPLORE_RATE_DECAY = .9
EPOCHS = 1000
GAMMA = .9

model = createModel(input_shape=(3,))
model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss='mean_squared_error')
modelClone = tf.keras.models.clone_model(model)

for epoch in range(EPOCHS):
  print('Start of %d epoch. Explore rate: %.3f' % (epoch, EXPLORE_RATE))
  # for stability
  modelClone.set_weights(model.get_weights())
  lossSum = 0
  for _ in range(TRAIN_EPISODES):
    states, actions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleBatch(
      batch_size=BATCH_SIZE, maxSamplesFromEpisode=16
    )
    actions = np.where(0 < actions, 1, 0)
    
    futureScores = modelClone.predict(nextStates).max(axis=1) * nextStateScoreMultiplier * GAMMA
    targets = modelClone.predict(states)
    targets[np.arange(len(targets)), actions] = rewards + futureScores

    lossSum += model.fit(states, targets, epochs=1, verbose=0).history['loss'][0]
  print('Avg. train loss: %.4f' % (lossSum / TRAIN_EPISODES))
  ##################
  print('Testing...')
  agent = DQNAgent(model, exploreRate=EXPLORE_RATE)
  testEnvs = [PendulumEnviroment() for _ in range(TEST_EPISODES)]

  for replay, isDone in emulateBatch(testEnvs, agent):
    memory.addEpisode(replay, terminated=not isDone)

  orderedScores = list(sorted([x.score for x in testEnvs], reverse=True))
  totalScores = sum(orderedScores) / TEST_EPISODES
  print('Avg. test score: %.1f' % (totalScores))
  metrics['scores']['avg.'].append(totalScores)
  metrics['scores']['top 10%'].append(orderedScores[int(TEST_EPISODES * 0.1)])
  metrics['scores']['top 50%'].append(orderedScores[int(TEST_EPISODES * 0.5)])
  metrics['scores']['top 90%'].append(orderedScores[int(TEST_EPISODES * 0.9)])
  ##################
  # debug
  if (epoch % 10) == 0:
    agent = DQNAgent(model, exploreRate=0)
    env.reset()
    while not env.done:
      action = agent.process(env.state)
      env.apply(action)
      env.render()
      time.sleep(.01)

  ##################
  EXPLORE_RATE = max((0.001, EXPLORE_RATE * EXPLORE_RATE_DECAY))
  plotData2file(metrics, 'chart.jpg')
