import tensorflow.keras as keras
import tensorflow.keras.layers as layers

def denseModel(input_shape, output_size, layersSize):
  inputs = res = layers.Input(shape=input_shape)
  for sz in layersSize:
    res = layers.Dense(sz, activation='relu')(res)

  return keras.Model(
    inputs=inputs,
    outputs=layers.Dense(output_size, activation='linear')(res)
  )

def createSimpleModel(input_shape, output_size=2):
  return denseModel(input_shape, output_size, [16, 8, 8, 4])

def createFatModel(input_shape, output_size=2):
  return denseModel(input_shape, output_size, [16, 16, 16, 16, output_size * 2])
