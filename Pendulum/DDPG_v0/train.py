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

from Pendulum.DDPG_v0.OUActionNoise import OUActionNoise
from Pendulum.DDPG_v0.CDDPGNetwork import CDDPGTrainable, CValueNetwork, CActorNetwork

from Utils.RawActionAgent import RawActionAgent
from Utils.ExperienceBuffers.CebLinear import CebLinear

from Pendulum import Utils
from Utils import plotData2file
from Pendulum.RawPendulumEnvironment import RawPendulumEnvironment
from tensorflow.keras import layers

def addUniformNoise(noisePower):
  def f(actions):
    noise = (np.random.random_sample(actions.shape) * 2.) - 1.
    return np.clip(actions + noise * noisePower, -1.0, 1.0)
  return f

def addNoise(noisePower):
  noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noisePower) * np.ones(1))
  def f(actions):
    return np.clip(actions + noise(), -1.0, 1.0)
  return f

"""
Build trainable model
"""
def buildActorModel(compile):
  states = layers.Input(shape=(3, ))
  out = layers.Dense(256, activation="relu")(states)
  out = layers.Dense(256, activation="relu")(out)
  out = layers.Dense(
    1, activation="tanh",
    kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
  )(out)

  model = tf.keras.Model(inputs=[states], outputs=[out])
  if compile:
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=None)
  return model

def buildCriticModel(compile):
  # State as input
  state_input = layers.Input(shape=(3,))
  state_out = layers.Dense(16, activation="relu")(state_input)
  state_out = layers.Dense(32, activation="relu")(state_out)

  # Action as input
  action_input = layers.Input(shape=(1,))
  action_out = layers.Dense(32, activation="relu")(action_input)

  # Both are passed through separate layer before concatenating
  concat = layers.Concatenate()([state_out, action_out])

  out = layers.Dense(256, activation="relu")(concat)
  out = layers.Dense(256, activation="relu")(out)
  outputs = layers.Dense(1)(out)

  model = tf.keras.Model([state_input, action_input], outputs)
  if compile:
    model.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss=None)
  return model

model = CDDPGTrainable(
  critic=CValueNetwork(
    createModel=buildCriticModel,
    loss=lambda y_true, y_pred: tf.math.reduce_mean(tf.math.square(y_true - y_pred))
  ),
  actor=CActorNetwork(createModel=buildActorModel)
)
##############
BATCH_SIZE = 64
GAMMA = 0.99
TRAIN_EPISODES = 200
TEST_EPISODES = 1
EPOCHS = 100
NOISE_STD = 0.1
NOISE_STD_DECAY = 0.99
TAU = 0.005

memory = CebLinear(maxSize=50000, sampleWeight='same')
metrics = {}
for epoch in range(EPOCHS):
  print('Start of %d epoch. Noise std: %.3f' % (epoch, NOISE_STD))
  ##################
  print('Testing...')
  scores = Utils.testAgent(
    RawActionAgent(model, processor=addNoise(NOISE_STD)),
    memory, TEST_EPISODES,
    env=RawPendulumEnvironment
  )
  Utils.trackScores(scores, metrics)
  ##################
  # train model
  lossesActor = []
  lossesCritic = []
  for _ in range(TRAIN_EPISODES):
    states, actions, rewards, nextStates, nextStateScoreMultiplier = memory.sampleBatch(BATCH_SIZE)
    nextStateScoreMultiplier = tf.convert_to_tensor(nextStateScoreMultiplier * GAMMA, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    
    lossCritic, lossActor = model.fit(states, actions, rewards, nextStates, nextStateScoreMultiplier)
    lossesCritic.append(lossCritic)
    lossesActor.append(lossActor)
    ######
    model.updateTargetModel(TAU)
    
  print('Avg. actor loss: %.4f' % (np.mean(lossesActor)))
  print('Avg. critic loss: %.4f' % (np.mean(lossesCritic)))
  ##################
  NOISE_STD = max((-0.001, NOISE_STD * NOISE_STD_DECAY))
  plotData2file(metrics, 'chart.jpg')