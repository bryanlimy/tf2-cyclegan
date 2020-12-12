from .registry import register

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from . import utils


@register('unet1')
def get_unet1(hparams, name=''):
  g = generator(
      hparams,
      output_channels=3,
      norm_type='instancenorm',
      name=utils.get_model_name(name, is_generator=True))
  d = discriminator(
      hparams,
      norm_type='instancenorm',
      name=utils.get_model_name(name, is_generator=False))
  return g, d


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      layers.Conv2D(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False))
  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(tfa.layers.InstanceNormalization())
  result.add(layers.LeakyReLU())
  return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      layers.Conv2DTranspose(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False))
  if norm_type.lower() == 'batchnorm':
    result.add(layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(tfa.layers.InstanceNormalization())
  if apply_dropout:
    result.add(layers.Dropout(0.5))
  result.add(layers.ReLU())
  return result


def generator(hparams, output_channels, norm_type='batchnorm',
              name='generator'):
  down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),
      downsample(128, 4, norm_type),
      downsample(256, 4, norm_type),
      downsample(512, 4, norm_type),
      downsample(512, 4, norm_type),
      downsample(512, 4, norm_type),
      downsample(512, 4, norm_type),
      downsample(512, 4, norm_type),
  ]
  up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),
      upsample(512, 4, norm_type, apply_dropout=True),
      upsample(512, 4, norm_type, apply_dropout=True),
      upsample(512, 4, norm_type),
      upsample(256, 4, norm_type),
      upsample(128, 4, norm_type),
      upsample(64, 4, norm_type),
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  concat = layers.Concatenate()

  inputs = layers.Input(shape=[None, None, 3])
  outputs = inputs

  skips = []
  for down in down_stack:
    outputs = down(outputs)
    skips.append(outputs)
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    outputs = up(outputs)
    outputs = concat([outputs, skip])

  outputs = layers.Conv2DTranspose(
      output_channels,
      4,
      strides=2,
      padding='same',
      kernel_initializer=initializer,
      activation='tanh')(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def discriminator(hparams, norm_type='batchnorm', name='discriminator'):
  assert norm_type in ['batchnorm', 'instancenorm']
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = layers.Input(shape=[None, None, 3], name='input_image')
  x = inp
  down1 = downsample(64, 4, norm_type, False)(x)
  down2 = downsample(128, 4, norm_type)(down1)
  down3 = downsample(256, 4, norm_type)(down2)

  zero_pad1 = layers.ZeroPadding2D()(down3)
  conv = layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)
  if norm_type == 'batchnorm':
    norm1 = layers.BatchNormalization()(conv)
  else:
    norm1 = tfa.layers.InstanceNormalization()(conv)
  leaky_relu = layers.LeakyReLU()(norm1)
  zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
  last = layers.Conv2D(
      1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
  return tf.keras.Model(inputs=inp, outputs=last, name=name)
