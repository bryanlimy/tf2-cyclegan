from .registry import register

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .utils import Activation, get_model_name


@register('mlp')
def get_mlp(hparams, name=''):
  g = generator(hparams, name=get_model_name(name, is_generator=True))
  d = discriminator(hparams, name=get_model_name(name, is_generator=False))
  return g, d


def generator(hparams, name='generator'):
  inputs = tf.keras.Input(shape=hparams.image_shape, name='inputs')

  outputs = layers.Flatten()(inputs)

  outputs = layers.Dense(hparams.num_units)(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(hparams.num_units)(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(hparams.num_units)(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(tf.math.reduce_prod(hparams.image_shape))(outputs)
  outputs = Activation('tanh')(outputs)
  outputs = layers.Reshape(target_shape=hparams.image_shape)(outputs)

  outputs = Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def discriminator(hparams, name='discriminator'):
  inputs = tf.keras.Input(hparams.image_shape, name='inputs')

  outputs = layers.Flatten()(inputs)

  outputs = layers.Dense(hparams.num_units)(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(hparams.num_units)(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(hparams.num_units)(outputs)
  outputs = Activation(hparams.activation)(outputs)
  outputs = layers.Dropout(hparams.dropout)(outputs)

  outputs = layers.Dense(1)(outputs)
  outputs = Activation('linear', dtype=tf.float32)(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
