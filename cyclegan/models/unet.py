from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from . import utils


@register('unet')
def get_unet(hparams, name=''):
  g = generator(hparams, name=utils.get_model_name(name, is_generator=True))
  d = discriminator(
      hparams, name=utils.get_model_name(name, is_generator=False))
  return g, d


def downsample(hparams, filters, kernel_size, initializer='glorot_uniform'):
  layer = tf.keras.Sequential()
  layer.add(
      layers.Conv2D(
          filters=filters,
          kernel_size=kernel_size,
          strides=2,
          padding='same',
          use_bias=False,
          kernel_initializer=initializer))
  layer.add(utils.Normalization(hparams.normalizer))
  layer.add(utils.Activation(hparams.activation))
  return layer


def upsample(hparams, filters, kernel_size, initializer='glorot_uniform'):
  layer = tf.keras.Sequential()
  layer.add(
      layers.Conv2DTranspose(
          filters=filters,
          kernel_size=kernel_size,
          strides=2,
          padding='same',
          use_bias=False,
          kernel_initializer=initializer))
  layer.add(utils.Normalization(hparams.normalizer))
  layer.add(utils.Activation(hparams.activation))
  if hparams.dropout > 0:
    layer.add(layers.Dropout(hparams.dropout))
  return layer


def generator(hparams, name='generator'):
  initializer = utils.Initializer(hparams.initializer)

  inputs = tf.keras.Input(shape=hparams.image_shape, name='inputs')
  outputs = inputs

  down_stack = [
      downsample(hparams, 64, 4, initializer=initializer),
      downsample(hparams, 128, 4, initializer=initializer),
      downsample(hparams, 256, 4, initializer=initializer),
      downsample(hparams, 512, 4, initializer=initializer),
      downsample(hparams, 512, 4, initializer=initializer),
      downsample(hparams, 512, 4, initializer=initializer),
      downsample(hparams, 512, 4, initializer=initializer),
      downsample(hparams, 512, 4, initializer=initializer),
  ]

  up_stack = [
      upsample(hparams, 512, 4, initializer=initializer),
      upsample(hparams, 512, 4, initializer=initializer),
      upsample(hparams, 512, 4, initializer=initializer),
      upsample(hparams, 512, 4, initializer=initializer),
      upsample(hparams, 256, 4, initializer=initializer),
      upsample(hparams, 128, 4, initializer=initializer),
      upsample(hparams, 64, 4, initializer=initializer),
  ]

  concat = tf.keras.layers.Concatenate()

  # down sample through the model
  skips = []
  for down in down_stack:
    outputs = down(outputs)
    skips.append(outputs)

  skips = reversed(skips[:-1])

  # up sample and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    outputs = up(outputs)
    outputs = concat([outputs, skip])

  outputs = layers.Conv2DTranspose(
      filters=3,
      kernel_size=4,
      strides=2,
      padding='same',
      kernel_initializer=initializer)(outputs)

  outputs = layers.Activation('tanh', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def discriminator(hparams, name='discriminator'):
  inputs = tf.keras.Input(hparams.image_shape, name='signals')

  initializer = utils.Initializer(hparams.initializer)

  outputs = downsample(hparams, 64, 4, initializer=initializer)(inputs)
  outputs = downsample(hparams, 128, 4, initializer=initializer)(outputs)
  outputs = downsample(hparams, 256, 4, initializer=initializer)(outputs)
  outputs = downsample(hparams, 512, 4, initializer=initializer)(outputs)

  outputs = layers.ZeroPadding2D()(outputs)
  outputs = layers.Conv2D(
      filters=512,
      kernel_size=4,
      strides=1,
      use_bias=False,
      kernel_initializer=initializer)(outputs)
  outputs = utils.Normalization(hparams.normalizer)(outputs)
  outputs = utils.Activation(hparams.activation)(outputs)
  outputs = layers.ZeroPadding2D()(outputs)
  outputs = layers.Conv2D(
      filters=1, kernel_size=4, strides=1,
      kernel_initializer=initializer)(outputs)

  if not hparams.patch_gan:
    outputs = layers.Flatten()(outputs)

  outputs = layers.Dense(1)(outputs)
  outputs = layers.Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
