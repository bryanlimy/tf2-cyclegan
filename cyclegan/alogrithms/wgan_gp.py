from .registry import register

import tensorflow as tf

from . import utils
from .gan import GAN
from .optimizer import Optimizer


@register('wgan_gp')
class WGAN_GP(GAN):

  def __init__(self, hparams, G, F, X, Y):
    super(WGAN_GP, self).__init__(hparams, G, F, X, Y)

    self.critic_steps = hparams.critic_steps
    self.gradient_penalty = hparams.gradient_penalty

  def generator_loss(self, discriminate_fake):
    return -tf.reduce_mean(discriminate_fake)

  def _train_generators(self, x, y):
    result = {}
    with tf.GradientTape(persistent=True) as tape:
      fake_x, fake_y, cycled_x, cycled_y = self.cycle_step(x, y, training=True)

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

      if self.mixed_precision:
        G_loss = self.G_optimizer.get_scaled_loss(G_loss)
        F_loss = self.F_optimizer.get_scaled_loss(F_loss)

    self.G_optimizer.update(self.G, G_loss, tape)
    self.F_optimizer.update(self.F, F_loss, tape)

    return result

  @staticmethod
  def _interpolation(real, fake):
    shape = (real.shape[0],) + (1,) * (len(real.shape) - 1)
    alpha = tf.random.uniform(shape, minval=0.0, maxval=1.0)
    return (alpha * real) + ((1 - alpha) * fake)

  def _gradient_penalty(self, discriminator, real, fake, training=False):
    interpolated = self._interpolation(real, fake)
    with tf.GradientTape() as tape:
      tape.watch(interpolated)
      discriminate_interpolated = discriminator(interpolated, training=training)
    gradient = tape.gradient(discriminate_interpolated, interpolated)
    norm = tf.norm(tf.reshape(gradient, shape=(gradient.shape[0], -1)), axis=1)
    return utils.mse(norm, 1.0)

  def discriminator_loss(self, discriminate_real, discriminate_fake):
    real_loss = -tf.reduce_mean(discriminate_real)
    fake_loss = tf.reduce_mean(discriminate_fake)
    return real_loss + fake_loss

  def _train_discriminators(self, x, y):
    result = {}
    with tf.GradientTape(persistent=True) as tape:
      fake_x, fake_y, cycled_x, cycled_y = self.cycle_step(x, y, training=True)

      discriminate_x = self.X(x, training=True)
      discriminate_y = self.Y(y, training=True)
      discriminate_fake_x = self.X(fake_x, training=True)
      discriminate_fake_y = self.Y(fake_y, training=True)

      # calculate loss for X
      X_loss = self.discriminator_loss(discriminate_x, discriminate_fake_x)
      X_gradient_penalty = self._gradient_penalty(
          self.X, real=x, fake=fake_x, training=True)
      result.update({
          'X_loss': X_loss,
          'X_gradient_penalty': X_gradient_penalty
      })
      X_loss += self.gradient_penalty * X_gradient_penalty

      # calculate loss for Y
      Y_loss = self.discriminator_loss(discriminate_y, discriminate_fake_y)
      Y_gradient_penalty = self._gradient_penalty(
          self.Y, real=y, fake=fake_y, training=True)
      result.update({
          'Y_loss': Y_loss,
          'Y_gradient_penalty': Y_gradient_penalty
      })
      Y_loss += self.gradient_penalty * Y_gradient_penalty

      if self.mixed_precision:
        X_loss = self.X_optimizer.get_scaled_loss(X_loss)
        Y_loss = self.Y_optimizer.get_scaled_loss(Y_loss)

    self.X_optimizer.update(self.X, X_loss, tape)
    self.Y_optimizer.update(self.Y, Y_loss, tape)

    return result

  @tf.function
  def train(self, x, y):
    result = {}
    for i in range(self.critic_steps):
      discriminator_results = self._train_discriminators(x, y)
      utils.update_dict(result, discriminator_results)
    generator_result = self._train_generators(x, y)
    utils.update_dict(result, generator_result)
    # calculate averages for all items in result
    return {key: tf.reduce_mean(value) for key, value in result.items()}
