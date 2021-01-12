import numpy as np
import tensorflow
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import random

# random network distillation curiosity model
# https://arxiv.org/abs/1810.12894
class CCuriosityIR:
  def __init__(self, maxCuriosity=1., layersSizes=None):
    self._model = None
    self._target = None
    self._maxCuriosity = maxCuriosity
    self._layersSizes = layersSizes
    return
  
  def fit(self, *inputs, **kwargs):
    inputs = self._process(inputs)
    target = self._target.predict(inputs)
    return self._model.fit(inputs, target, epochs=1, verbose=0, **kwargs)
  
  def rewards(self, *inputs):
    inputs = self._process(inputs)
    predicted = self._model.predict(inputs)
    target = self._target.predict(inputs)
    
    return np.clip(
      np.sqrt(np.power(target - predicted, 2).sum(axis=-1)),
      a_min=0,
      a_max=self._maxCuriosity
    )
  
  def _process(self, inputs):
    res = np.concatenate(
      [ x.reshape((x.shape[0], -1)) for x in inputs ],
      axis=-1
    )
    if self._model is None:
      self._model = self._createModel(res.shape[-1])
      self._target = tensorflow.keras.models.clone_model(self._model) # random weights?
    return res
  
  def _createModel(self, size):
    # just random dense network
    inputs = res = layers.Input(shape=(size,))
    layersSizes = [
      int(size * (.5 + random.random())) for _ in range(random.randint(2, 8))
    ] if self._layersSizes is None else self._layersSizes
    for sz in layersSizes:
      res = layers.Dense(sz, activation='tanh')(res)
  
    model = keras.Model(inputs=inputs, outputs=res)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss="mse")
    return model
