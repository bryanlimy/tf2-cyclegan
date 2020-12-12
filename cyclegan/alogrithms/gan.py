from .registry import register

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

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

    # self.G_optimizer = Optimizer(hparams)
    # self.F_optimizer = Optimizer(hparams)
    # self.X_optimizer = Optimizer(hparams)
    # self.Y_optimizer = Optimizer(hparams)
    self.G_optimizer = Adam(hparams.learning_rate, beta_1=0.5)
    self.F_optimizer = Adam(hparams.learning_rate, beta_1=0.5)
    self.X_optimizer = Adam(hparams.learning_rate, beta_1=0.5)
    self.Y_optimizer = Adam(hparams.learning_rate, beta_1=0.5)

    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    self.alpha = hparams.alpha
    self.beta = hparams.beta

    self.mixed_precision = hparams.mixed_precision
    self.error_function = utils.get_error_function(hparams.error)

  def generator_loss(self, discriminate_fake):
    return self.cross_entropy(
        tf.ones_like(discriminate_fake), discriminate_fake)

  def discriminator_loss(self, discriminate_real, discriminate_fake):
    real_loss = self.cross_entropy(
        tf.ones_like(discriminate_real), discriminate_real)
    fake_loss = self.cross_entropy(
        tf.zeros_like(discriminate_fake), discriminate_fake)
    return 0.5 * (real_loss + fake_loss)

  def cycle_loss(self, real, cycled):
    return 10.0 * self.error_function(real, cycled)

  def identity_loss(self, real, identity):
    """ calculate identity loss || x - F(x) || or || y - G(y) || """
    return 0.5 * 10.0 * self.error_function(real, identity)

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
    with tf.GradientTape(persistent=True) as tape:
      fake_y = self.G(x, training=True)
      cycled_x = self.F(fake_y, training=True)

      fake_x = self.F(y, training=True)
      cycled_y = self.G(fake_x, training=True)

      discriminate_x = self.X(x, training=True)
      discriminate_y = self.Y(y, training=True)
      discriminate_fake_x = self.X(fake_x, training=True)
      discriminate_fake_y = self.Y(fake_y, training=True)

      G_loss = self.generator_loss(discriminate_fake_y)
      F_loss = self.generator_loss(discriminate_fake_x)

      cycle_loss = self.cycle_loss(x, cycled_x) + self.cycle_loss(y, cycled_y)

      # calculate identity loss
      same_y = self.G(y, training=True)
      same_x = self.F(x, training=True)
      G_identity_loss = self.identity_loss(y, same_y)
      F_identity_loss = self.identity_loss(x, same_x)

      total_G_loss = G_loss + cycle_loss + G_identity_loss
      total_F_loss = F_loss + cycle_loss + F_identity_loss

      X_loss = self.discriminator_loss(discriminate_x, discriminate_fake_x)
      Y_loss = self.discriminator_loss(discriminate_y, discriminate_fake_y)

    G_gradient = tape.gradient(total_G_loss, self.G.trainable_variables)
    F_gradient = tape.gradient(total_F_loss, self.F.trainable_variables)
    X_gradient = tape.gradient(X_loss, self.X.trainable_variables)
    Y_gradient = tape.gradient(Y_loss, self.Y.trainable_variables)

    self.G_optimizer.apply_gradients(
        zip(G_gradient, self.G.trainable_variables))
    self.F_optimizer.apply_gradients(
        zip(F_gradient, self.F.trainable_variables))
    self.X_optimizer.apply_gradients(
        zip(X_gradient, self.X.trainable_variables))
    self.Y_optimizer.apply_gradients(
        zip(Y_gradient, self.Y.trainable_variables))

    return {
        'G_loss': G_loss,
        'F_loss': F_loss,
        'cycle_loss': cycle_loss,
        'G_identity_loss': G_identity_loss,
        'F_identity_loss': F_identity_loss,
        'X_loss': X_loss,
        'Y_loss': Y_loss
    }

  @tf.function
  def validate(self, x, y):
    fake_y = self.G(x, training=False)
    cycled_x = self.F(fake_y, training=False)

    fake_x = self.F(y, training=False)
    cycled_y = self.G(fake_x, training=False)

    same_y = self.G(y, training=False)
    same_x = self.F(x, training=False)

    return {
        'MSE(X, F(G(X)))': utils.mean_square_error(x, cycled_x),
        'MSE(Y, G(F(Y)))': utils.mean_square_error(y, cycled_y),
        'MSE(X, F(X))': utils.mean_square_error(x, same_x),
        'MSE(Y, G(Y))': utils.mean_square_error(y, same_y)
    }
