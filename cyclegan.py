import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from cyclegan.utils import utils
from cyclegan.utils.summary_helper import Summary

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
LAMBDA = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


def random_crop(image):
  cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image


def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image


def random_jitter(image):
  image = tf.image.resize(
      image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = random_crop(image)
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


def preprocess_image_test(image, label):
  image = normalize(image)
  return image


def get_dataset(hparams):
  dataset, metadata = tfds.load(
      'cycle_gan/horse2zebra', with_info=True, as_supervised=True)

  train_horses, train_zebras = dataset['trainA'], dataset['trainB']
  test_horses, test_zebras = dataset['testA'], dataset['testB']

  train_horses = train_horses.map(
      preprocess_image_train,
      num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(
          hparams.batch_size)

  train_zebras = train_zebras.map(
      preprocess_image_train,
      num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(
          hparams.batch_size)

  test_horses = test_horses.map(
      preprocess_image_test,
      num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(
          hparams.batch_size)

  test_zebras = test_zebras.map(
      preprocess_image_test,
      num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(
          hparams.batch_size)

  return train_horses, train_zebras, test_horses, test_zebras


def generate_images(summary, generator_g, generator_f, real_horses, real_zebras,
                    epoch):
  fake_zebras = generator_g(real_horses, training=False)
  cycled_horses = generator_f(fake_zebras, training=False)

  fake_horses = generator_f(real_zebras, training=False)
  cycled_zebras = generator_g(fake_horses, training=False)

  for i in range(3):
    summary.plot_transformation(
        f'X cycle/image_{i + 1:02d}',
        images=[
            real_horses[i, ...],
            fake_zebras[i, ...],
            cycled_horses[i, ...],
        ],
        labels=['X', 'G(X)', 'F(G(X))'],
        step=epoch,
        training=False)

    summary.plot_transformation(
        f'Y cycle/image_{i + 1:02d}',
        images=[
            real_zebras[i, ...],
            fake_horses[i, ...],
            cycled_zebras[i, ...],
        ],
        labels=['Y', 'F(Y)', 'G(F(Y))'],
        step=epoch,
        training=False)


def update_dict(dict1, dict2, replace=False):
  """ update dict1 with the items in dict2 """
  for key, value in dict2.items():
    if replace:
      dict1[key] = value
    else:
      if key not in dict1:
        dict1[key] = []
      dict1[key].append(value)


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False))
  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(tfa.layers.InstanceNormalization())
  result.add(tf.keras.layers.LeakyReLU())
  return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(
          filters,
          size,
          strides=2,
          padding='same',
          kernel_initializer=initializer,
          use_bias=False))
  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(tfa.layers.InstanceNormalization())
  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result


def unet_generator(output_channels, norm_type='batchnorm'):
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
  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, 3])
  outputs = inputs

  skips = []
  for down in down_stack:
    outputs = down(outputs)
    skips.append(outputs)
  skips = reversed(skips[:-1])
  for up, skip in zip(up_stack, skips):
    outputs = up(outputs)
    outputs = concat([outputs, skip])

  outputs = tf.keras.layers.Conv2DTranspose(
      output_channels,
      4,
      strides=2,
      padding='same',
      kernel_initializer=initializer,
      activation='tanh')(outputs)

  return tf.keras.Model(inputs=inputs, outputs=outputs)


def discriminator(norm_type='batchnorm'):
  assert norm_type in ['batchnorm', 'instancenorm']
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  x = inp
  down1 = downsample(64, 4, norm_type, False)(x)
  down2 = downsample(128, 4, norm_type)(down1)
  down3 = downsample(256, 4, norm_type)(down2)
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
  conv = tf.keras.layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)
  if norm_type == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  else:
    norm1 = tfa.layers.InstanceNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last = tf.keras.layers.Conv2D(
      1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
  return tf.keras.Model(inputs=inp, outputs=last)


loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def mean_square_error(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_true - y_pred))


def discriminator_loss(real, generated):
  real_loss = loss_function(tf.ones_like(real), real)
  generated_loss = loss_function(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5


def generator_loss(generated):
  return loss_function(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
  return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image):
  return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


@tf.function
def train_step(
    real_x,
    real_y,
    generator_g,
    generator_f,
    discriminator_x,
    discriminator_y,
    generator_g_optimizer,
    generator_f_optimizer,
    discriminator_x_optimizer,
    discriminator_y_optimizer,
):
  with tf.GradientTape(persistent=True) as tape:
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_y = generator_g(real_y, training=True)
    same_x = generator_f(real_x, training=True)

    g_identity_loss = identity_loss(real_y, same_y)
    f_identity_loss = identity_loss(real_x, same_x)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(
        real_y, cycled_y)

    total_gen_g_loss = gen_g_loss + cycle_loss + g_identity_loss
    total_gen_f_loss = gen_f_loss + cycle_loss + f_identity_loss

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(
      zip(generator_g_gradients, generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(
      zip(generator_f_gradients, generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(
      zip(discriminator_x_gradients, discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(
      zip(discriminator_y_gradients, discriminator_y.trainable_variables))

  return {
      'G_loss': gen_g_loss,
      'F_loss': gen_f_loss,
      'X_loss': disc_x_loss,
      'Y_loss': disc_y_loss,
      'cycle_loss': cycle_loss,
      'G_identity_loss': g_identity_loss,
      'F_identity_loss': f_identity_loss
  }


@tf.function
def validation_step(real_x, real_y, generator_g, generator_f):
  fake_y = generator_g(real_x, training=False)
  cycled_x = generator_f(fake_y, training=False)

  fake_x = generator_f(real_y, training=False)
  cycled_y = generator_g(fake_x, training=False)

  same_y = generator_g(real_y, training=False)
  same_x = generator_f(real_x, training=False)

  return {
      'MSE(X, F(G(X)))': mean_square_error(real_x, cycled_x),
      'MSE(Y, G(F(Y)))': mean_square_error(real_y, cycled_y),
      'MSE(X, F(X))': mean_square_error(real_x, same_x),
      'MSE(Y, G(Y))': mean_square_error(real_y, same_y)
  }


def main(hparams):
  train_horses, train_zebras, test_horses, test_zebras = get_dataset(hparams)

  num_train_steps = int(tf.data.experimental.cardinality(train_horses))
  num_val_steps = int(tf.data.experimental.cardinality(test_horses))

  sample_horse = next(iter(train_horses))
  sample_zebra = next(iter(train_zebras))

  generator_g = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
  generator_f = unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

  discriminator_x = discriminator(norm_type='instancenorm')
  discriminator_y = discriminator(norm_type='instancenorm')

  generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  summary = Summary(hparams)

  for epoch in range(hparams.epochs):
    print(f'Epoch {epoch+1:03d}/{hparams.epochs:03d}')
    train_metrics, val_metrics = {}, {}
    start = time.time()
    for image_x, image_y in tqdm(
        tf.data.Dataset.zip((train_horses, train_zebras)),
        desc='Train',
        total=num_train_steps):
      log = train_step(
          image_x,
          image_y,
          generator_g,
          generator_f,
          discriminator_x,
          discriminator_y,
          generator_g_optimizer,
          generator_f_optimizer,
          discriminator_x_optimizer,
          discriminator_y_optimizer,
      )
      update_dict(train_metrics, log, replace=False)
    for image_x, image_y in tqdm(
        tf.data.Dataset.zip((test_horses, test_zebras)),
        desc='Validation',
        total=num_val_steps):
      log = validation_step(image_x, image_y, generator_g, generator_f)
      update_dict(val_metrics, log, replace=False)
    end = time.time()

    for key, values in train_metrics.items():
      summary.scalar(
          f'loss/{key}', tf.reduce_mean(values), epoch, training=True)
    for key, values in val_metrics.items():
      summary.scalar(
          f'mse/{key}', tf.reduce_mean(values), epoch, training=False)

    print(f'MSE(X, F(G(X))): {np.mean(val_metrics["MSE(X, F(G(X)))"]):.04f}\t\t'
          f'MSE(X, F(X)): {np.mean(val_metrics["MSE(Y, G(F(Y)))"]):.04f}\n'
          f'MSE(Y, G(F(Y))): {np.mean(val_metrics["MSE(X, F(X))"]):.04f}\t\t'
          f'MSE(Y, G(Y)): {np.mean(val_metrics["MSE(Y, G(Y))"]):.04f}\n'
          f'Elapse: {(end - start) / 60:.02f} mins\n')

    if epoch % 10 == 0 or epoch == hparams.epochs - 1:
      generate_images(summary, generator_g, generator_f, sample_horse,
                      sample_zebra, epoch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--epochs', default=200, type=int)
  parser.add_argument('--batch_size', default=16, type=int)
  params = parser.parse_args()

  np.random.seed(1234)
  tf.random.set_seed(1234)

  main(params)
