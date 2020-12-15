from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam


class Optimizer:

  def __init__(self, hparams, model):
    self.model = model
    self.mixed_precision = hparams.mixed_precision
    self.optimizer = Adam(hparams.learning_rate, beta_1=0.5)
    if self.mixed_precision:
      self.optimizer = mixed_precision.LossScaleOptimizer(
          self.optimizer, dynamic=True)

  def trainable_variables(self):
    return self.model.trainable_variables

  def get_scaled_loss(self, loss):
    return self.optimizer.get_scaled_loss(loss)

  def get_unscaled_gradients(self, scaled_gradients):
    return self.optimizer.get_unscaled_gradients(scaled_gradients)

  def minimize(self, loss, tape):
    gradients = tape.gradient(loss, self.trainable_variables)
    if self.mixed_precision:
      gradients = self.get_unscaled_gradients(gradients)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
