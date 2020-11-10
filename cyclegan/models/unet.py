from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import Conv1DTranspose, Activation, PhaseShuffle, get_model_name


@register('unet')
def get_unet(hparams, name=''):
  g = generator(hparams, name=get_model_name(name, is_generator=True))
  d = discriminator(hparams, name=get_model_name(name, is_generator=False))
  return g, d


def downsample(hparams, filters, kernel_size):
  initializer = tf.random_normal_initializer(0., 0.02)
  layer = tf.keras.Sequential()
  layer.add(
      layers.Conv1D(
          filters=filters,
          kernel_size=kernel_size,
          strides=2,
          padding='same',
          use_bias=False,
          kernel_initializer=initializer))
  if hparams.layer_norm:
    layer.add(layers.LayerNormalization())
  layer.add(Activation(hparams.activation))
  return layer


def upsample(hparams, filters, kernel_size):
  initializer = tf.random_normal_initializer(0., 0.02)
  layer = tf.keras.Sequential()
  layer.add(
      Conv1DTranspose(
          filters=filters,
          kernel_size=kernel_size,
          strides=2,
          padding='same',
          use_bias=False,
          kernel_initializer=initializer))
  layer.add(layers.LayerNormalization())
  if hparams.dropout > 0:
    layer.add(layers.Dropout(hparams.dropout))
  layer.add(Activation(hparams.activation))
  return layer


def generator(hparams, name='generator'):
  inputs = tf.keras.Input(shape=hparams.signal_shape, name='inputs')
  outputs = inputs

  down_stack = [
      downsample(hparams, 64, 4),
      downsample(hparams, 128, 4),
      downsample(hparams, 256, 4),
      downsample(hparams, 512, 4),
      downsample(hparams, 512, 4),
      downsample(hparams, 512, 4),
      downsample(hparams, 512, 4),
      downsample(hparams, 512, 4),
  ]

  up_stack = [
      upsample(hparams, 512, 4),
      upsample(hparams, 512, 4),
      upsample(hparams, 512, 4),
      upsample(hparams, 512, 4),
      upsample(hparams, 256, 4),
      upsample(hparams, 128, 4),
      upsample(hparams, 64, 4),
  ]

  concat = tf.keras.layers.Concatenate()

  # Downsampling through the model
  skips = []
  for down in down_stack:
    outputs = down(outputs)
    skips.append(outputs)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    outputs = up(outputs)
    outputs = concat([outputs, skip])

  outputs = Conv1DTranspose(
      filters=hparams.num_neurons,
      kernel_size=4,
      strides=2,
      padding='same',
      kernel_initializer=tf.random_normal_initializer(0., 0.02),
      activation=None)(outputs)

  outputs = layers.Dense(hparams.num_neurons)(outputs)
  outputs = layers.Activation(
      'sigmoid' if hparams.normalize else 'linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def discriminator(hparams, name='discriminator'):
  inputs = tf.keras.Input(hparams.signal_shape, name='signals')

  initializer = tf.random_normal_initializer(0., 0.02)

  outputs = downsample(hparams, 64, 4)(inputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)
  outputs = downsample(hparams, 128, 4)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)
  outputs = downsample(hparams, 256, 4)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)
  outputs = downsample(hparams, 512, 4)(outputs)
  outputs = PhaseShuffle(outputs.shape, m=hparams.m)(outputs)

  outputs = layers.ZeroPadding1D()(outputs)
  outputs = layers.Conv1D(
      filters=512,
      kernel_size=4,
      strides=1,
      use_bias=False,
      kernel_initializer=initializer)(outputs)

  outputs = layers.LayerNormalization()(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.ZeroPadding1D()(outputs)

  outputs = layers.Conv1D(
      filters=1, kernel_size=4, strides=1,
      kernel_initializer=initializer)(outputs)

  if not hparams.patch_gan:
    outputs = layers.Flatten()(outputs)

  outputs = layers.Dense(1)(outputs)
  outputs = layers.Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
