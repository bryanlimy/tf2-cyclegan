import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def Activation(name, **kwargs):
  return layers.LeakyReLU(
      **kwargs) if name == 'lrelu' else layers.Activation(name, **kwargs)


class Conv1DTranspose(layers.Layer):

  def __init__(self,
               filters,
               kernel_size,
               strides,
               padding='same',
               output_padding=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               activation=layers.LeakyReLU()):
    super().__init__()
    self.activation = activation

    self.conv2dtranspose = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(kernel_size, 1),
        strides=(strides, 1),
        padding=padding,
        output_padding=output_padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
    )

  def call(self, inputs):
    outputs = tf.expand_dims(inputs, axis=2)
    outputs = self.conv2dtranspose(outputs)
    outputs = tf.squeeze(outputs, axis=2)
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs


class PhaseShuffle(layers.Layer):
  ''' Phase Shuffle introduced in the WaveGAN paper so that the discriminator 
  are less sensitive toward periodic patterns which occurs quite frequently in
  signal data '''

  def __init__(self, input_shape, m=0, mode='reflect'):
    super().__init__()
    self.shape = input_shape
    self.m = m
    self.mode = mode

  def call(self, inputs, **kwargs):
    if self.m == 0:
      return inputs

    w = self.shape[1]

    # shift on the temporal dimension
    shift = tf.random.uniform([],
                              minval=-self.m,
                              maxval=self.m + 1,
                              dtype=tf.int32)

    if shift > 0:
      # shift to the right
      paddings = [[0, 0], [0, shift], [0, 0]]
      start, end = shift, w + shift
    else:
      # shift to the left
      paddings = [[0, 0], [tf.math.abs(shift), 0], [0, 0]]
      start, end = 0, w

    outputs = tf.pad(inputs, paddings=paddings, mode=self.mode)
    outputs = outputs[:, start:end, :]
    return tf.ensure_shape(outputs, shape=self.shape)


def count_trainable_params(model):
  ''' return the number of trainable parameters'''
  return np.sum(
      [tf.keras.backend.count_params(p) for p in model.trainable_weights])


def get_model_name(name, is_generator):
  if is_generator:
    return f'generator_{name}'
  else:
    return f'discriminator_{"Y" if name == "G" else "X"}'


def model_summary(hparams, model):
  ''' get tf.keras model summary as a string and save it as txt '''
  stream = io.StringIO()
  model.summary(print_fn=lambda x: stream.write(x + '\n'))
  summary = stream.getvalue()
  stream.close()

  with open(os.path.join(hparams.output_dir, f'{model.name}.txt'), 'a') as file:
    file.write(summary)

  return summary
