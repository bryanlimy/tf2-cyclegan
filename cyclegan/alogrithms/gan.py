from .registry import register

import tensorflow as tf

from . import utils
from .optimizer import Optimizer


@register('gan')
class GAN:

  def __init__(self, hparams, G, F, X, Y):
    """ 
      G(x): x -> y
      F(y): y -> x
    """
    self.G = G
    self.F = F
    self.X = X
    self.Y = Y

    self.G_optimizer = Optimizer(hparams)
    self.F_optimizer = Optimizer(hparams)
    self.X_optimizer = Optimizer(hparams)
    self.Y_optimizer = Optimizer(hparams)

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    self.discriminator_loss_coefficient = 0.5
    self.cycle_loss_coefficient = 10

    self.mixed_precision = hparams.mixed_precision
    self.error = utils.mse if hparams.cycle_error == 'mse' else utils.mae

  def get_models(self):
    return self.G, self.F, self.X, self.Y

  def generator_loss(self, discriminate_fake):
    return self.cross_entropy(
        tf.ones_like(discriminate_fake), discriminate_fake)

  def discriminator_loss(self, discriminate_real, discriminate_fake):
    real_loss = self.cross_entropy(
        tf.ones_like(discriminate_real), discriminate_real)
    fake_loss = self.cross_entropy(
        tf.zeros_like(discriminate_fake), discriminate_fake)
    total_loss = real_loss + fake_loss
    return self.discriminator_loss_coefficient * total_loss

  def cycle_loss(self, real, cycled):
    return self.cycle_loss_coefficient * self.error(real, cycled)

  def identity_loss(self, real, identity):
    """ Calculate the MAE(x, F(x)) or MAE(y, G(y))"""
    return self.cycle_loss_coefficient * 0.5 * self.error(real, identity)

  @tf.function
  def cycle_step(self, x, y, training=False):
    # x -> fake y -> cycled x
    fake_y = self.G(x, training=training)
    cycled_x = self.F(fake_y, training=training)

    # y -> fake x -> cycled y
    fake_x = self.F(y, training=training)
    cycled_y = self.G(fake_x, training=training)

    return fake_x, fake_y, cycled_x, cycled_y

  @tf.function
  def train(self, x, y):
    result = {}
    with tf.GradientTape(persistent=True) as tape:
      fake_x, fake_y, cycled_x, cycled_y = self.cycle_step(x, y, training=True)

      discriminate_x = self.X(x, training=True)
      discriminate_y = self.Y(y, training=True)
      discriminate_fake_x = self.X(fake_x, training=True)
      discriminate_fake_y = self.Y(fake_y, training=True)

      G_loss = self.generator_loss(discriminate_fake_y)
      F_loss = self.generator_loss(discriminate_fake_x)

      result.update({'G_loss': G_loss, 'F_loss': F_loss})

      cycle_loss = self.cycle_loss(x, cycled_x) + self.cycle_loss(y, cycled_y)

      result.update({'cycle_loss': cycle_loss})

      x_identity_loss = self.identity_loss(x, self.F(x, training=True))
      y_identity_loss = self.identity_loss(y, self.G(y, training=True))

      result.update({
          'x_identity_loss': x_identity_loss,
          'y_identity_loss': y_identity_loss
      })

      G_loss += cycle_loss + x_identity_loss
      F_loss += cycle_loss + y_identity_loss

      X_loss = self.discriminator_loss(discriminate_x, discriminate_fake_x)
      Y_loss = self.discriminator_loss(discriminate_y, discriminate_fake_y)

      result.update({'X_loss': X_loss, 'Y_loss': Y_loss})

      if self.mixed_precision:
        G_loss = self.G_optimizer.get_scaled_loss(G_loss)
        F_loss = self.F_optimizer.get_scaled_loss(F_loss)
        X_loss = self.X_optimizer.get_scaled_loss(X_loss)
        Y_loss = self.Y_optimizer.get_scaled_loss(Y_loss)

    self.G_optimizer.update(self.G, G_loss, tape)
    self.F_optimizer.update(self.F, F_loss, tape)
    self.X_optimizer.update(self.X, X_loss, tape)
    self.Y_optimizer.update(self.Y, Y_loss, tape)

    return result

  @tf.function
  def validate(self, x, y):
    _, _, cycled_x, cycled_y = self.cycle_step(x, y, training=False)

    same_x = self.F(x, training=False)
    same_y = self.G(y, training=False)

    return {
        'MSE(X, F(G(X)))': utils.mse(x, cycled_x),
        'MSE(Y, G(F(Y)))': utils.mse(y, cycled_y),
        'MSE(X, F(X))': utils.mse(x, same_x),
        'MSE(Y, G(Y))': utils.mse(y, same_y)
    }
