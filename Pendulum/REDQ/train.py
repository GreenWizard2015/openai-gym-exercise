# -*- coding: utf-8 -*-
import sys
import os
import tensorflow as tf

COLAB_ENV = 'COLAB_GPU' in os.environ
if COLAB_ENV:
  # fix resolve modules
  from os.path import dirname
  sys.path.append(dirname(dirname(dirname(__file__))))
else: # local GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4 * 1024)]
  )

import numpy as np

from Pendulum.PendulumEnvironment import PendulumEnvironment
from Pendulum.REDQ.CREDQEnsemble import CREDQEnsembleTrainable

from Utils.ExperienceBuffers.CebPrioritized import CebPrioritized
from Utils import emulate, plotData2file
from Utils.DQNAgent import DQNAgent
from Utils.RandomAgent import RandomAgent
from Utils.MappedActions import MappedActions
from Pendulum import Utils, DQNModels

BATCH_SIZE = 1024
TRAIN_EPISODES = 500
TEST_EPISODES = 256
EXPLORE_RATE = .5
EXPLORE_RATE_DECAY = .9
EPOCHS = 100
GAMMA = .9
ACTIONS = MappedActions(N=9, valuesRange=(-1, 1))

N = 3
M = 2

model = CREDQEnsembleTrainable({
  'shape': (3, ),
  'N models': N,
  'M estimators': M,
  'submodel': lambda: DQNModels.createDuelingModel(
    input_shape=(3,), duelingInnerLayerSize=16, output_size=ACTIONS.N
  ),
  'optimizer': lambda: tf.optimizers.Adam(lr=1e-5, clipnorm=1.0),
  'micro batch': 64,
})
model.summary()

env = PendulumEnvironment()
memory = CebPrioritized(maxSize=5000, sampleWeight='abs') # from dueling dqn
# collect random experience
agent = RandomAgent(low=-1, high=1)
for episodeN in range(1000):
  replay, done = emulate(env, agent)
  memory.addEpisode(replay, terminated=not done)
print('random experience collected')

metrics = {}
exploreRate = EXPLORE_RATE
trackedScores = []
for epoch in range(EPOCHS):
  print('[N = %d, M = %d] Start of %d epoch. Explore rate: %.3f' % (N, M, epoch, exploreRate))
  # train model
  model.updateTargetModel()
  losses = []
  for _ in range(TRAIN_EPISODES):
    states, actions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleBatch(
      batch_size=BATCH_SIZE, maxSamplesFromEpisode=16
    )
    actions = ACTIONS.toIndex(actions)
    
    _, loss = model.fit(states, actions, rewards, nextStates, nextStateScoreMultiplier)
    losses.append(loss)
    ######
    
  print('Avg. train loss: %.4f' % (np.mean(losses)))
  ##################
  print('Testing...')
  scores = Utils.testAgent(
    DQNAgent(model, actions=ACTIONS, exploreRate=exploreRate),
    memory, TEST_EPISODES
  )
  Utils.trackScores(scores, metrics)
  ##################
  if ((epoch % 10) == 0) and not COLAB_ENV: # debug
    Utils.showAgentPlay( DQNAgent(model, actions=ACTIONS, exploreRate=0) )
  ##################
  exploreRate = max((0.001, exploreRate * EXPLORE_RATE_DECAY))
  plotData2file(metrics, 'chart-%d-%d.jpg' % (N, M, ))