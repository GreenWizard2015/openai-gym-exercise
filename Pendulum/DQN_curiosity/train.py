# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy
# limit GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2 * 1024)]
)

from Pendulum.PendulumEnviroment import PendulumEnviroment
from Pendulum.DQNModels import createFatModel

from Utils import emulate, plotData2file
import Utils.ExperienceBuffers as EB
from Utils.DQNAgent import DQNAgent
from Utils.RandomAgent import RandomAgent
from Utils.MappedActions import MappedActions

from Pendulum.DQN_curiosity.trainingStage import train
from Pendulum import Utils

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def createCuriousityModel():
  inputs = res = layers.Input(shape=(4))
  for sz in [10, 10, 10, 10, 10, 10]:
    res = layers.Dense(sz, activation='relu')(res)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Dense(3, activation='tanh')(res)
  )

metrics = {}

env = PendulumEnviroment()
memory = EB.CebPrioritized(maxSize=5000, sampleWeight='abs')
# collect random experience
agent = RandomAgent(low=-1, high=1)
for episodeN in range(100):
  replay, done = emulate(env, agent)
  memory.addEpisode(replay, terminated=not done)
print('random experience collected')
#######################################
# train agent
BATCH_SIZE = 512
TRAIN_EPISODES = 25
TEST_EPISODES = 256
EXPLORE_RATE = .005
EXPLORE_RATE_DECAY = .95
EPOCHS = 1000
GAMMA = .9
ACTIONS = MappedActions(N=9, valuesRange=(-1, 1))

model = createFatModel(input_shape=(3,), output_size=ACTIONS.N)
model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss='mean_squared_error')

curiousityModel = createCuriousityModel()
curiousityModel.compile(optimizer=tf.optimizers.Adam(lr=1e-3), loss='mean_squared_error')

BOOTSTRAPPED_STEPS = 10
for epoch in range(EPOCHS):
  print('Start of %d epoch. Explore rate: %.3f' % (epoch, EXPLORE_RATE))
  ##################
  # Training
  trainLoss = train(
    model, memory, curiousityModel,
    {
      'gamma': GAMMA,
      'actions': ACTIONS,
      'batchSize': BATCH_SIZE,
      'episodes': TRAIN_EPISODES,
      'steps': BOOTSTRAPPED_STEPS,
      
      'curiousity update rate': 5,
      'curiousity range': BOOTSTRAPPED_STEPS,
      'curiousity function': lambda V, x: V + numpy.abs(V * x.sum(axis=-1)),
    }
  )
  print('Avg. train loss: %.4f' % trainLoss)
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
