# -*- coding: utf-8 -*-
import tensorflow as tf
import math
from Utils.Networks.GhostNetwork import GhostNetwork
# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2 * 1024)]
)

import numpy as np
from Pendulum.PendulumEnvironment import PendulumEnvironment
from Pendulum.DQNModels import createFatModel

from Utils import plotData2file
import Utils.ExperienceBuffers as EB
from Utils.DQNAgent import DQNAgent
from Utils.RandomAgent import RandomAgent
from Utils.MappedActions import MappedActions

from Pendulum.DQN_curiosity.trainingStage import train, trainCuriosity
from Pendulum import Utils
from Utils.CuriosityModel.CCuriosityIR import CCuriosityIR
from Utils.CuriosityModel.CCuriosityIRWatched import CCuriosityIRWatched
###############
def replayProcessor(curiosity, rewardScale, normalize):
  maxScore = [-np.inf]
  def f(replay):
    states, actions, rewards, nextStates = zip(*replay)
    rewards = np.array(rewards)
    if normalize:
      maxScore[0] = mxs = max((maxScore[0], np.abs(rewards).max()))
      rewards /= mxs # normalize to -1..1
 
    IR = curiosity.rewards(np.array(states), np.array(actions))
    rewards += IR * rewardScale
    return list(zip(states, actions, rewards, nextStates))
  return f
###############
BATCH_SIZE = 512
TRAIN_EPISODES = 25
CURIOSITY_TRAIN_EPISODES = 10
TEST_EPISODES = 256
EXPLORE_RATE = .05
EXPLORE_RATE_DECAY = .95
EPOCHS = 1000
GAMMA = .9
ACTIONS = MappedActions(N=9, valuesRange=(-1, 1))
STEPS_PER_EPISODE = 200
BOOTSTRAPPED_STEPS = 10

metrics = {}

env = PendulumEnvironment()
memory = EB.CebLinear(maxSize=10 * TEST_EPISODES * STEPS_PER_EPISODE, sampleWeight='abs')
curiosityModel = CCuriosityIRWatched( CCuriosityIR(layersSizes=[10, 10, 10]) )
processor = replayProcessor(curiosityModel, rewardScale=1.0/BOOTSTRAPPED_STEPS, normalize=True)
# collect random experience
for episodeN in range(2):
  Utils.testAgent(
    RandomAgent(low=-1, high=1),
    memory, episodes=100,
    processor=processor
  )
print('random experience collected')
####################
model = createFatModel(input_shape=(3,), output_size=ACTIONS.N)
model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.Huber(delta=1.0))

ghostNetwork = GhostNetwork(model, mixer='hard')
for epoch in range(EPOCHS):
  print('Start of %d epoch. Explore rate: %.3f' % (epoch, EXPLORE_RATE))
  ##################
  # Training
  ghostNetwork.update()
  trainLoss = train(
    ghostNetwork, memory,
    {
      'gamma': GAMMA,
      'actions': ACTIONS,
      'batchSize': BATCH_SIZE,
      'episodes': TRAIN_EPISODES,
      'steps': BOOTSTRAPPED_STEPS,
    }
  )
  print('Avg. train loss: %.4f' % trainLoss)
  trainCuriosity(
    curiosityModel, memory,
    {
      'batchSize': BATCH_SIZE,
      'episodes': CURIOSITY_TRAIN_EPISODES,
    }
  )
  ##################
  print('Testing...')
  curiosityModel.reset()
  scores = Utils.testAgent(
    DQNAgent(model, actions=ACTIONS, exploreRate=EXPLORE_RATE),
    memory, TEST_EPISODES,
    processor=processor
  )
  print(curiosityModel.info)
  Utils.trackScores(scores, metrics)
  ##################
  if (epoch % 10) == 0: # debug
    Utils.showAgentPlay( DQNAgent(model, actions=ACTIONS, exploreRate=0) )
  ##################
  EXPLORE_RATE = max((0.001, EXPLORE_RATE * EXPLORE_RATE_DECAY))
  plotData2file(metrics, 'chart.jpg')
