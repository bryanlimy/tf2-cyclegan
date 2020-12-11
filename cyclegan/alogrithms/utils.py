import tensorflow as tf


def mean_square_error(a, b):
  return tf.reduce_mean(tf.square(a - b))


def mean_absolute_error(a, b):
  return tf.reduce_mean(tf.abs(a - b))


def huber_loss(a, b):
  return tf.reduce_mean(tf.losses.huber(a, b, delta=1.0))


def get_error_function(name):
  if name == 'mse':
    return mean_square_error
  elif name == 'mae':
    return mean_absolute_error
  elif name == 'huber':
    return huber_loss
