import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

class LazyNetwork:
  def __init__(self, model, batchSize, patience, fitArgs, loss=None):
    self._model = model
    self._batchSize = batchSize
    self._patience = patience
    
    loss = model.compiled_loss if loss is None else loss
    self._loss = lambda ytrue, ypred: K.eval(loss(ytrue, ypred))
    self._fitArgs = fitArgs
    
    self.resetStats()
    self._dropSamples()
    return
  
  def _dropSamples(self):
    self._samplesX = np.array([])
    self._samplesY = np.array([])
    self._samplesLoss = np.array([])
    return
  
  def resetStats(self):
    self.stats = {
      'total samples': 0,
      'trained samples': 0,
      'losses': [], 
    }
    return
  
  def predict(self, inputs):
    return self._model.predict(inputs)
  
  def _learn(self):
    def softmax(x):
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=0)
    
    batchInd = np.random.choice(
      self._samplesX.shape[0],
      size=self._batchSize,
      replace=False,
      p=softmax(self._samplesLoss)
    )
    
    self.stats['trained samples'] += len(batchInd)
    self.stats['losses'].append(
      self._model.fit(
        self._samplesX[batchInd],
        self._samplesY[batchInd],
        **self._fitArgs
      ).history['loss'][0]
    )
    
    self._dropSamples()
    return
  
  def fit(self, X, Y):
    self.stats['total samples'] += len(X)
    
    predicted = self._model.predict(X)
    losses = self._loss(
      tf.convert_to_tensor(Y, tf.float32),
      tf.convert_to_tensor(predicted, tf.float32)
    )
    
    if 0 < self._samplesX.shape[0]:
      self._samplesX = np.concatenate((self._samplesX, X))
      self._samplesY = np.concatenate((self._samplesY, Y))
      self._samplesLoss = np.concatenate((self._samplesLoss, losses))
    else:
      self._samplesX = X
      self._samplesY = Y
      self._samplesLoss = losses

    if self._patience <= self._samplesX.shape[0]:
      self._learn()
    return
  
  def __enter__(self):
    return self
  
  def __exit__(self, *args):
    if self._batchSize <= self._samplesX.shape[0]:
      self._learn()
    return