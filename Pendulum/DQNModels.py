import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def denseModel(input_shape, output_size, layersSize, outputActivation):
  inputs = res = layers.Input(shape=input_shape)
  for sz in layersSize:
    res = layers.Dense(sz, activation='relu')(res)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Dense(output_size, activation=outputActivation)(res)
  )

def createSimpleModel(input_shape, output_size=2, outputActivation='linear'):
  return denseModel(input_shape, output_size, [16, 8, 8, 4], outputActivation)

def createFatModel(input_shape, output_size=2, outputActivation='linear'):
  return denseModel(input_shape, output_size, [16, 16, 16, 16, output_size * 2], outputActivation)

def createDuelingModel(input_shape, duelingInnerLayerSize, output_size):
  layersSize = [16, 16, output_size * 2]

  inputs = res = layers.Input(shape=input_shape)
  for sz in layersSize:
    res = layers.Dense(sz, activation='relu')(res)

  valueBranch = layers.Dense(duelingInnerLayerSize, activation='relu')(res)
  valueBranch = layers.Dense(1, activation='linear')(valueBranch)
  
  actionsBranch = layers.Dense(duelingInnerLayerSize, activation='relu')(res)
  actionsBranch = layers.Dense(output_size, activation='linear')(actionsBranch)
  
  res = layers.Lambda(
    lambda x: x[1] + (x[0] - tf.reduce_mean(x[0], axis=-1, keepdims=True))
  )([actionsBranch, valueBranch])
  
  return keras.Model(inputs=inputs, outputs=res)