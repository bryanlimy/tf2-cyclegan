import tensorflow as tf


def mse(a, b):
  return tf.reduce_mean(tf.square(a - b))


def mae(a, b):
  return tf.reduce_mean(tf.abs(a - b))


def denormalize(x, x_min, x_max):
  ''' re-scale signals back to its original range '''
  return x * (x_max - x_min) + x_min


def update_dict(dict1, dict2):
  """ Add content of dict2 to dict1 """
  for key, value in dict2.items():
    if key not in dict1:
      dict1[key] = []
    dict1[key].append(value)
