import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Optimizer:

  def __init__(self, hparams):
    self.mixed_precision = hparams.mixed_precision
    self.optimizer = Adam(hparams.learning_rate, beta_1=0.5)
    if self.mixed_precision:
      self.optimizer = mixed_precision.LossScaleOptimizer(
          self.optimizer, loss_scale='dynamic')

  def get_scaled_loss(self, loss):
    return self.optimizer.get_scaled_loss(loss)

  def get_unscaled_gradients(self, scaled_gradients):
    return self.optimizer.get_unscaled_gradients(scaled_gradients)

  def update(self, model, loss, tape):
    gradients = tape.gradient(loss, model.trainable_variables)
    if self.mixed_precision:
      gradients = self.get_unscaled_gradients(gradients)
    self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
