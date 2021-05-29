"""
Generator and discriminator architectures in CycleGAN
Reference: https://keras.io/examples/generative/cyclegan/#building-blocks-used-in-the-cyclegan-generators-and-discriminators
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class ReflectionPadding2D(layers.Layer):
  """ Implements Reflection Padding as a layer.
  
  Args:
    padding (tuple): Amount of padding for the spatial dimensions (H, W).
  Returns:
    A padded tensor with the same type as the input tensor.
  """

  def __init__(self, paddings: tuple = (1, 1), **kwargs):
    self.paddings = [
        [0, 0],
        [paddings[0], paddings[0]],
        [paddings[1], paddings[1]],
        [0, 0],
    ]
    super(ReflectionPadding2D, self).__init__(**kwargs)

  def call(self, inputs, **kwargs):
    return tf.pad(inputs, paddings=self.paddings, mode="REFLECT")


def residual_block(
    inputs,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
  filters = inputs.shape[-1]
  outputs = inputs

  outputs = ReflectionPadding2D()(outputs)
  outputs = layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      kernel_initializer=kernel_initializer,
      padding=padding,
      use_bias=use_bias,
  )(outputs)
  outputs = tfa.layers.InstanceNormalization(
      gamma_initializer=gamma_initializer)(outputs)
  outputs = activation(outputs)

  outputs = ReflectionPadding2D()(outputs)
  outputs = layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      kernel_initializer=kernel_initializer,
      padding=padding,
      use_bias=use_bias,
  )(outputs)
  outputs = tfa.layers.InstanceNormalization(
      gamma_initializer=gamma_initializer)(outputs)
  outputs = layers.add([inputs, outputs])
  return outputs


def downsample(
    inputs,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
  outputs = layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      kernel_initializer=kernel_initializer,
      padding=padding,
      use_bias=use_bias,
  )(inputs)
  outputs = tfa.layers.InstanceNormalization(
      gamma_initializer=gamma_initializer)(outputs)
  if activation is not None:
    outputs = activation(outputs)
  return outputs


def upsample(
    inputs,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
  outputs = layers.Conv2DTranspose(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      kernel_initializer=kernel_initializer,
      use_bias=use_bias,
  )(inputs)
  outputs = tfa.layers.InstanceNormalization(
      gamma_initializer=gamma_initializer)(outputs)
  if activation is not None:
    outputs = activation(outputs)
  return outputs


def get_generator(input_shape: tuple,
                  filters: int = 64,
                  num_downsampling_blocks: int = 2,
                  num_residual_blocks: int = 9,
                  num_upsample_blocks: int = 2,
                  gamma_initializer=gamma_init,
                  name: str = 'generator'):
  inputs = layers.Input(shape=input_shape, name='inputs')

  outputs = ReflectionPadding2D(padding=(3, 3))(inputs)
  outputs = layers.Conv2D(filters=filters,
                          kernel_size=(7, 7),
                          kernel_initializer=kernel_init,
                          use_bias=False)(outputs)
  outputs = tfa.layers.InstanceNormalization(
      gamma_initializer=gamma_initializer)(outputs)
  outputs = layers.Activation("relu")(outputs)

  # Downsampling
  for _ in range(num_downsampling_blocks):
    filters *= 2
    outputs = downsample(outputs,
                         filters=filters,
                         activation=layers.Activation("relu"))

  # Residual blocks
  for _ in range(num_residual_blocks):
    outputs = residual_block(outputs, activation=layers.Activation("relu"))

  # Upsampling
  for _ in range(num_upsample_blocks):
    filters //= 2
    outputs = upsample(outputs, filters, activation=layers.Activation("relu"))

  # Final block
  outputs = ReflectionPadding2D(paddings=(3, 3))(outputs)
  outputs = layers.Conv2D(filters=3, kernel_size=(7, 7),
                          padding="valid")(outputs)
  outputs = layers.Activation("tanh")(outputs)

  return tf.keras.models.Model(inputs, outputs, name=name)


def get_discriminator(input_shape: tuple,
                      filters: int = 64,
                      num_downsampling: int = 3,
                      kernel_initializer=kernel_init,
                      name: str = 'discriminator'):
  inputs = layers.Input(shape=input_shape, name='inputs')

  outputs = layers.Conv2D(
      filters=filters,
      kernel_size=(4, 4),
      strides=(2, 2),
      padding="same",
      kernel_initializer=kernel_initializer,
  )(inputs)
  outputs = layers.LeakyReLU(0.2)(outputs)

  for i in range(num_downsampling):
    filters *= 2
    if i < 2:
      outputs = downsample(
          outputs,
          filters=filters,
          activation=layers.LeakyReLU(0.2),
          kernel_size=(4, 4),
          strides=(2, 2),
      )
    else:
      outputs = downsample(
          outputs,
          filters=filters,
          activation=layers.LeakyReLU(0.2),
          kernel_size=(4, 4),
          strides=(1, 1),
      )

  outputs = layers.Conv2D(filters=1,
                          kernel_size=(4, 4),
                          strides=(1, 1),
                          padding="same",
                          kernel_initializer=kernel_initializer)(outputs)

  return tf.keras.models.Model(inputs, outputs, name=name)
