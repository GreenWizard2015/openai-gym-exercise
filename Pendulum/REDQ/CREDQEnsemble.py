import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def combineQ(W, stackedQ):
  alter = (1.0 - W) * stackedQ.dtype.max
  masked = (stackedQ * W[:, :, None]) + alter[:, :, None]
  return tf.reduce_min(masked, axis=-2)

def createEnsemble(params, compileModel):
  N_models = params['N models']
  submodel = params['submodel']
  state = layers.Input(shape=params['shape'])
  submodelsW = layers.Input(shape=(N_models, )) # for randomness during training
  
  submodels = [submodel() for _ in range(N_models)]
  submodelsQ = [x(state) for x in submodels]
  
  combined = combineQ(submodelsW, tf.stack(submodelsQ, axis=1))
  
  model = tf.keras.Model(
    inputs=[state, submodelsW],
    outputs=[combined, *submodelsQ]
  )
  
  if compileModel:
    for x in submodels:
      x.compile(optimizer=params['optimizer'](), loss=None)
  return model

def createTrainStep(model, targetModel, minibatchSize=64):
  submodels = [x for x in model.layers if x.name.startswith('model')]
  
  @tf.function
  def _trainBatch(states, actions, rewards, nextStates, nextStateScoreMultiplier, nextQScore):
    allW = tf.ones((tf.shape(states)[0], len(submodels)))
    # Double DQN next reward
    nextAction = tf.argmax(model([nextStates, allW], training=False)[0], axis=-1)
    futureScores = tf.gather(nextQScore, nextAction, batch_dims=1)
    nextRewards = rewards + futureScores * nextStateScoreMultiplier
    
    with tf.GradientTape(persistent=True) as tape:
      predictions = model([states, allW])
      ########
      targets = predictions[0]
      targets = (nextRewards[:, None] * actions) + (targets * (1.0 - actions))
      ########
      losses = [
        tf.reduce_mean(tf.keras.losses.huber(targets, x)) for x in predictions[1:]
      ]
      
    for submodel in submodels:
      grads = tape.gradient(losses, submodel.trainable_weights)
      submodel.optimizer.apply_gradients(zip(grads, submodel.trainable_weights))
    
    return tf.reduce_mean(losses), nextRewards

  @tf.function
  def step(states, actions, rewards, nextStates, nextStateScoreMultiplier, W):
    rewards = tf.cast(rewards, tf.float32)
    nextStateScoreMultiplier = tf.cast(nextStateScoreMultiplier, tf.float32)
    nextQScore = targetModel([nextStates, W], training=False)[0]
    actions = tf.one_hot(actions, tf.shape(nextQScore)[-1])
    
    loss = 0.0
    nextRewards = tf.zeros((tf.shape(W)[0],))
    indices = tf.reshape(tf.range(minibatchSize), (-1, 1))
    for i in tf.range(0, tf.shape(states)[0], minibatchSize):
      bLoss, NR = _trainBatch(
        states[i:i+minibatchSize],
        actions[i:i+minibatchSize],
        rewards[i:i+minibatchSize],
        nextStates[i:i+minibatchSize],
        nextStateScoreMultiplier[i:i+minibatchSize],
        nextQScore[i:i+minibatchSize]
      )
      nextRewards = tf.tensor_scatter_nd_update(nextRewards, i + indices, NR)
      loss += bLoss

    newValues = model([nextStates, tf.ones_like(W)], training=False)[0]
    errors = tf.keras.losses.huber(nextRewards, tf.reduce_sum(actions * newValues, axis=-1))
    return errors, loss
  
  return step

###########
class CREDQEnsemble:
  def __init__(self, params, train=False, model=None):
    self._N = params['N models']
    self._model = model
    if not model:
      self._model = createEnsemble(params, compileModel=train) 
    return
  
  def predict(self, X):
    W = np.ones((len(X), self._N))
    return self._model([X, W])[0]
  
  def __call__(self, X):
    return self.predict(X)
  
  def load(self, filepath):
    self._model.load_weights(filepath)
    return
    
  def save(self, filepath):
    self._model.save_weights(filepath)
    return
  
  def summary(self):
    self._model.summary()
    return
###########
class CREDQEnsembleTrainable(CREDQEnsemble):
  def __init__(self, params):
    super().__init__(params, train=True)
    self._M = params['M estimators']
    
    self._targetModel = createEnsemble(params, compileModel=False)
    self.updateTargetModel()
    
    self._trainStep = createTrainStep(
      self._model, self._targetModel,
      minibatchSize=params.get('micro batch', 64)
    ) 
    return
  
  def fit(self, states, actions, rewards, nextStates, nextStateScoreMultiplier):
    # select M random models per sample (because we can)
    W = np.zeros((len(states), self._N))
    for i in range(len(states)):
      ind = np.random.choice(self._N, self._M, replace=False)
      W[i, ind] = 1.0
      
    return self._trainStep(states, actions, rewards, nextStates, nextStateScoreMultiplier, W)
  
  def updateTargetModel(self):
    self._targetModel.set_weights(self._model.get_weights())
    return