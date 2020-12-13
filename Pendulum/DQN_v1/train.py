# -*- coding: utf-8 -*-
import tensorflow as tf
# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4 * 1024)]
)

import numpy as np

from Pendulum.PendulumEnvironment import PendulumEnvironment
from Pendulum.DQNModels import createSimpleModel

from Utils import emulate, plotData2file
from Utils.ExperienceBuffers.CebPrioritized import CebPrioritized
from Utils.DQNAgent import DQNAgent
from Utils.RandomAgent import RandomAgent
from Utils.MappedActions import MappedActions
from Pendulum import Utils

metrics = {}

env = PendulumEnvironment()
memory = CebPrioritized(maxSize=5000, sampleWeight='abs')
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
ACTIONS = MappedActions(N=2, valuesRange=(-1, 1))

model = createSimpleModel(input_shape=(3,), output_size=ACTIONS.N)
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
    actions = ACTIONS.toIndex(actions)
    
    futureScores = modelClone.predict(nextStates).max(axis=-1) * nextStateScoreMultiplier
    targets = modelClone.predict(states)
    targets[np.arange(len(targets)), actions] = rewards + futureScores * GAMMA

    lossSum += model.fit(states, targets, epochs=1, verbose=0).history['loss'][0]
  print('Avg. train loss: %.4f' % (lossSum / TRAIN_EPISODES))
  ##################
  print('Testing...')
  scores = Utils.testAgent(
    DQNAgent(model, actions=ACTIONS, exploreRate=EXPLORE_RATE),
    memory, TEST_EPISODES
  )
  Utils.trackScores(scores, metrics)
  ##################
  if (epoch % 10) == 0: # debug
    Utils.showAgentPlay( DQNAgent(model, actions=ACTIONS, exploreRate=0) )
  ##################
  EXPLORE_RATE = max((0.001, EXPLORE_RATE * EXPLORE_RATE_DECAY))
  plotData2file(metrics, 'chart.jpg')
