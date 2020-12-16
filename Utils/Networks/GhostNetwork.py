import tensorflow as tf
import numpy as np

def WeightsLinearMix(tau):
  def f(old, new):
    old = np.array(old)
    new = np.array(new)
    return old + (new - old) * tau

  return f

class GhostNetwork:
  def __init__(self, mainNetwork, mixer=None):
    self._main = mainNetwork
    self._ghost = tf.keras.models.clone_model(mainNetwork)
    self._mixer = mixer
    self.update('hard')
    return 
  
  def update(self, mixer=None):
    mixer = self._mixer if mixer is None else mixer
    if mixer is None:
      raise Exception('Can\'t mix weight without mixer.')
    
    if 'hard' == mixer:
      self._ghost.set_weights(self._main.get_weights())
      return

    self._ghost.set_weights(mixer(
      self._ghost.get_weights(),
      self._main.get_weights()
    ))
    return
  
  def predict(self, inputs):
    return self._ghost.predict(inputs)
  
  def fit(self, *args, **kwargs):
    return self._main.fit(*args, **kwargs)
  
  def evaluate(self, *args, **kwargs):
    return self._main.evaluate(*args, **kwargs)
  
  def weights(self):
    return self._ghost.get_weights()