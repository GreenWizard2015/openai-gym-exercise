import tensorflow as tf

class CDoubleNetwork:
  def __init__(self, createModel):
    self._model = createModel(compile=True)
    self._targetModel = createModel(compile=False)
    self.updateTargetModel()
    return
  
  @tf.function
  def _emaUpdate(self, targetV, srcV, tau):
    for (a, b) in zip(targetV, srcV):
      a.assign(b * tau + a * (1 - tau))
    return
  
  def updateTargetModel(self, tau=1.0):
    if 1.0 <= tau:
      self._targetModel.set_weights(self._model.get_weights())
    else:
      self._emaUpdate(self._targetModel.trainable_variables, self._model.trainable_variables, tau)
    return
###################
class CValueNetwork(CDoubleNetwork): # critic
  def __init__(self, createModel, loss):
    super().__init__(createModel)
    self._loss = loss
    return
  
  @tf.function
  def fit(self, states, actions, rewards, nextStates, nextActions, nextRewardMultiplier):
    with tf.GradientTape() as tape:
      nextSAValue = self._targetModel([nextStates, nextActions], training=False)
      targetValues = rewards + nextRewardMultiplier[:, None] * nextSAValue
      
      predictedV = self._model([states, actions], training=True)
      
      tf.assert_equal(tf.shape(targetValues), tf.shape(predictedV))
      loss = self._loss(targetValues, predictedV)

    TV = self._model.trainable_variables
    grads = tape.gradient(loss, TV)
    self._model.optimizer.apply_gradients(zip(grads, TV))
    return loss
  
  @tf.function
  def __call__(self, StatesActions, training=False):
    return self._model(StatesActions, training=training)
  
###################
class CActorNetwork(CDoubleNetwork):
  def __init__(self, createModel):
    super().__init__(createModel)
    return
  
  @tf.function  
  def fit(self, states, critic):
    with tf.GradientTape() as tape:
      actions = self._model([states], training=True)
      values = critic([states, actions], training=False)
      loss = -tf.reduce_mean(values)

    TV = self._model.trainable_variables
    grads = tape.gradient(loss, TV)
    self._model.optimizer.apply_gradients(zip(grads, TV))
    return loss
  
  @tf.function
  def __call__(self, states, training=False):
    return self._model([states], training=training)

  @tf.function
  def target(self, states, training=False):
    return self._targetModel([states], training=training)
###################
class CDDPGTrainable:
  def __init__(self, actor, critic):
    self.actor = actor
    self.critic = critic
    return
  
  @tf.function
  def fit(self, states, actions, rewards, nextStates, nextRewardMultiplier):
    criticLoss = self.fitCritic(states, actions, rewards, nextStates, nextRewardMultiplier)
    actorLoss = self.fitActor(states)
    return criticLoss, actorLoss
  
  @tf.function
  def fitCritic(self, states, actions, rewards, nextStates, nextRewardMultiplier):
    nextActions = self.actor.target(nextStates, training=False)
    return self.critic.fit(states, actions, rewards, nextStates, nextActions, nextRewardMultiplier)
  
  @tf.function
  def fitActor(self, states):
    return self.actor.fit(states, self.critic)
  
  def updateTargetModel(self, tau=1.0):
    self.actor.updateTargetModel(tau)
    self.critic.updateTargetModel(tau)
    return

  @tf.function
  def __call__(self, states):
    return self.actor(states)