# -*- coding: utf-8 -*-
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
  gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3 * 1024)]
)


import numpy as np

from Pendulum.PendulumEnvironment import PendulumEnvironment
from Pendulum.DQNModels import createFatModel

from Pendulum.REINFORCE.ReinforceAgent import ReinforceAgent

from Utils import plotData2file
from Utils.MappedActions import MappedActions
from Pendulum import Utils
from tensorflow import keras

def complexLoss(actions, rewards, y_pred):
  y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0) # log(0) => nan
  actions = tf.one_hot(actions, 2) # tf.gather(y_pred, actions) => error
  values = tf.reduce_sum(y_pred * actions)
  return tf.math.log(values) * rewards * -1.0

def wrapForTraining(agent):
  inputA = keras.layers.Input(shape=agent.input_shape[1:])
  inputAction = keras.layers.Input(shape=(1, ), dtype=tf.int32)
  inputReward = keras.layers.Input(shape=(1, ))
  
  res = agent(inputA)
  
  model = keras.Model(inputs=[inputA, inputAction, inputReward], outputs=[res])
  model.add_loss(complexLoss(inputAction, inputReward, res))
  model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss=None)
  return model

def discountedReturns(rewards, discount):
  returns = []
  total = 0.0
  for r in rewards[::-1]:
    val = r + discount * total
    returns.insert(0, val)
    total = val
  return returns

metrics = {}
#######################################
# train agent
TEST_EPISODES_PER_EPOCH = 25
EXPLORE_RATE = .01
EXPLORE_RATE_DECAY = .999
EPOCHS = 10000
GAMMA = .95
ACTIONS = MappedActions(N=2, valuesRange=(-1, 1))

model = createFatModel(input_shape=(3,), output_size=ACTIONS.N, outputActivation='softmax')
trainable = wrapForTraining(model)

env = PendulumEnvironment(seed=123) # fixed seed
for epoch in range(EPOCHS):
  print('Start of %d epoch. Explore rate: %.3f' % (epoch, EXPLORE_RATE))
  ##################
  scores = []
  agent = ReinforceAgent(model, actions=ACTIONS, noise=EXPLORE_RATE)
  for _ in range(TEST_EPISODES_PER_EPOCH):
    env.reset()
    agent.reset()
    replay = []
    done = False
    while not done:
      action = agent.process(env.state)
      _, reward, done, prevState = env.apply(action)
      replay.append((prevState, action, reward))
    ##
    states, actions, rewards = zip( *replay )
    
    actions = ACTIONS.toIndex(actions)
    trainable.fit(
      [
        np.array(states),
        np.array(actions),
        np.array(discountedReturns(rewards, GAMMA))
      ],
      epochs=1, verbose=0
    )
    ##
    scores.append(env.score)

  Utils.trackScores(scores, metrics)
  ##################
  EXPLORE_RATE = max((0.001, EXPLORE_RATE * EXPLORE_RATE_DECAY))
  plotData2file(metrics, 'chart.jpg')