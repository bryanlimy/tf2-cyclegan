import os
import re
import argparse
import typing as t
import numpy as np
from math import ceil
from tqdm import tqdm
from glob import glob
from time import time
import tensorflow as tf
from shutil import rmtree
import tensorflow_datasets as tfds

from cyclegan import utils, model

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SHAPE = (286, 286)
INPUT_SHAPE = (256, 256, 3)


def get_datasets(
    args,
    strategy: tf.distribute.Strategy,
    buffer_size: int = 256) -> t.Tuple[tf.data.Dataset, tf.data.Dataset]:
  """ Load and return preprocessed and distributed horse2zebra dataset """
  dataset, metadata = tfds.load("cycle_gan/horse2zebra",
                                with_info=True,
                                as_supervised=True)
  train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
  test_horses, test_zebras = dataset["testA"], dataset["testB"]

  # calculate the number of train and test steps needed per epoch
  get_size = lambda name: metadata.splits.get(name).num_examples
  num_train_samples = min([get_size('trainA'), get_size('trainB')])
  num_test_samples = min([get_size('testA'), get_size('testB')])
  args.train_steps = ceil(num_train_samples / args.global_batch_size)
  args.test_steps = ceil(num_test_samples / args.global_batch_size)

  def normalize_image(image):
    """ normalize image to [-1, 1] """
    image = tf.cast(image, dtype=tf.float32)
    return (image / 127.5) - 1.0

  def preprocess_train(image, _):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, size=IMAGE_SHAPE)
    image = tf.image.random_crop(image, size=INPUT_SHAPE)
    image = normalize_image(image)
    return image

  def preprocess_test(image, _):
    image = tf.image.resize(image, size=INPUT_SHAPE[:2])
    image = normalize_image(image)
    return image

  train_horses = train_horses.map(preprocess_train, num_parallel_calls=AUTOTUNE)
  train_horses = train_horses.cache()
  train_horses = train_horses.shuffle(buffer_size)
  train_horses = train_horses.batch(args.batch_size)

  train_zebras = train_zebras.map(preprocess_train, num_parallel_calls=AUTOTUNE)
  train_zebras = train_zebras.cache()
  train_zebras = train_zebras.shuffle(buffer_size)
  train_zebras = train_zebras.batch(args.batch_size)

  test_horses = test_horses.map(preprocess_test, num_parallel_calls=AUTOTUNE)
  test_horses = test_horses.cache()
  sample_horses = test_horses.take(5).batch(1)
  test_horses = test_horses.batch(args.batch_size)

  test_zebras = test_zebras.map(preprocess_test, num_parallel_calls=AUTOTUNE)
  test_zebras = test_zebras.cache()
  sample_zebras = test_zebras.take(5).batch(1)
  test_zebras = test_zebras.batch(args.batch_size)

  train_ds = tf.data.Dataset.zip((train_horses, train_zebras))
  test_ds = tf.data.Dataset.zip((test_horses, test_zebras))
  # take 5 samples from the test set for plotting
  sample_ds = tf.data.Dataset.zip((sample_horses, sample_zebras))

  # create distributed datasets
  train_ds = strategy.experimental_distribute_dataset(train_ds)
  test_ds = strategy.experimental_distribute_dataset(test_ds)

  return train_ds, test_ds, sample_ds


def MAE(y_true, y_pred):
  """ return the per sample mean absolute error """
  outputs = tf.abs(y_true - y_pred)
  return tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))


def MSE(y_true, y_pred):
  """ return the per sample mean squared error """
  outputs = tf.square(y_true - y_pred)
  return tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))


def BCE(y_true, y_pred, from_logits: bool = False):
  """ return the per sample binary cross entropy """
  outputs = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_true, axis=-1),
                                                tf.expand_dims(y_pred, axis=-1),
                                                from_logits=from_logits)
  return tf.reduce_mean(outputs, axis=list(range(1, len(outputs.shape))))


class CycleGAN:
  """
  CycleGAN class that support per replica loss calculation
  Reference: https://keras.io/examples/generative/cyclegan/#build-the-cyclegan-model
  """

  def __init__(self, args, strategy: tf.distribute.Strategy):
    self.strategy = strategy
    self.global_batch_size = args.global_batch_size
    # cycle consistency loss coefficient
    self.lambda_cycle = 10.0
    # identity loss coefficient
    self.lambda_identity = 0.5 * self.lambda_cycle

    # create checkpoint directory
    self.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)

    with self.strategy.scope():
      # initialize models
      self.G = model.get_generator(input_shape=INPUT_SHAPE, name='gen_G')
      self.F = model.get_generator(input_shape=INPUT_SHAPE, name='gen_F')
      self.X = model.get_discriminator(input_shape=INPUT_SHAPE, name='dis_X')
      self.Y = model.get_discriminator(input_shape=INPUT_SHAPE, name='dis_Y')

      # initialize optimizers
      self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                  beta_1=0.5,
                                                  beta_2=0.9)
      self.F_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                  beta_1=0.5,
                                                  beta_2=0.9)
      self.X_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                  beta_1=0.5,
                                                  beta_2=0.9)
      self.Y_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                                  beta_1=0.5,
                                                  beta_2=0.9)

      # initialize checkpoint
      self.checkpoint = tf.train.Checkpoint(G=self.G,
                                            F=self.F,
                                            X=self.X,
                                            Y=self.Y,
                                            G_optimizer=self.G_optimizer,
                                            F_optimizer=self.F_optimizer,
                                            X_optimizer=self.X_optimizer,
                                            Y_optimizer=self.Y_optimizer)

  def save_checkpoint(self, epoch: int):
    """ save checkpoint to checkpoint_dir """
    file_prefix = os.path.join(self.checkpoint_dir, f'epoch-{epoch:03d}')
    self.checkpoint.write(file_prefix)
    print(f'saved checkpoint to {file_prefix}')

  def load_checkpoint(self, expect_partial: bool = False):
    """
    load latest checkpoint from checkpoint_dir and return checkpoint epoch
    """
    epoch = -1
    checkpoints = glob(os.path.join(self.checkpoint_dir, 'epoch-*'))
    if checkpoints:
      last_checkpoint = sorted(checkpoints)[-1]
      if expect_partial:
        self.checkpoint.read(last_checkpoint).expected_partial()
      else:
        self.checkpoint.read(last_checkpoint)
      # get checkpoint epoch
      matches = re.match(r"epoch-(\d{3}).", os.path.basename(last_checkpoint))
      epoch = int(matches.groups()[0])
      print(f'loaded checkpoint from {last_checkpoint}')
    return epoch

  def reduce_mean(self, inputs):
    """ return inputs mean with respect to the global_batch_size """
    return tf.reduce_sum(inputs) / self.global_batch_size

  def generator_loss(self, discriminate_fake):
    per_sample_loss = MSE(y_true=tf.ones_like(discriminate_fake),
                          y_pred=discriminate_fake)
    return self.reduce_mean(per_sample_loss)

  def cycle_loss(self, real_samples, cycle_samples):
    per_sample_loss = MAE(y_true=real_samples, y_pred=cycle_samples)
    return self.lambda_cycle * self.reduce_mean(per_sample_loss)

  def identity_loss(self, real_samples, identity_samples):
    per_sample_loss = MAE(y_true=real_samples, y_pred=identity_samples)
    return self.lambda_identity * self.reduce_mean(per_sample_loss)

  def discriminator_loss(self, discriminate_real, discriminate_fake):
    real_loss = MSE(y_true=tf.ones_like(discriminate_real),
                    y_pred=discriminate_real)
    fake_loss = MSE(y_true=tf.zeros_like(discriminate_fake),
                    y_pred=discriminate_fake)
    per_sample_loss = 0.5 * (real_loss + fake_loss)
    return self.reduce_mean(per_sample_loss)

  @tf.function
  def cycle_step(self, x, y, training: bool = False):
    # x -> fake y -> cycle x
    fake_y = self.G(x, training=training)
    cycle_x = self.F(fake_y, training=training)
    # y -> fake x -> cycle y
    fake_x = self.F(y, training=training)
    cycle_y = self.G(fake_x, training=training)
    return fake_x, fake_y, cycle_x, cycle_y

  def train_step(self, x, y):
    result = {}
    with tf.GradientTape(persistent=True) as tape:
      fake_y = self.G(x, training=True)
      fake_x = self.F(y, training=True)

      discriminate_fake_x = self.X(fake_x, training=True)
      discriminate_fake_y = self.Y(fake_y, training=True)

      G_loss = self.generator_loss(discriminate_fake_y)
      F_loss = self.generator_loss(discriminate_fake_x)

      G_cycle_loss = self.cycle_loss(y, self.G(fake_x, training=True))
      F_cycle_loss = self.cycle_loss(x, self.F(fake_y, training=True))

      G_identity_loss = self.identity_loss(y, self.G(y, training=True))
      F_identity_loss = self.identity_loss(x, self.F(x, training=True))

      G_total_loss = G_loss + G_cycle_loss + G_identity_loss
      F_total_loss = F_loss + F_cycle_loss + F_identity_loss

      result.update({
          'loss_G/loss': G_loss,
          'loss_G/cycle': G_cycle_loss,
          'loss_G/identity': G_identity_loss,
          'loss_G/total': G_total_loss,
          'loss_F/loss': F_loss,
          'loss_F/cycle': F_cycle_loss,
          'loss_F/identity': F_identity_loss,
          'loss_F/total': F_total_loss
      })

      discriminate_x = self.X(x, training=True)
      discriminate_y = self.Y(y, training=True)
      discriminate_fake_x = self.X(fake_x, training=True)
      discriminate_fake_y = self.Y(fake_y, training=True)

      X_loss = self.discriminator_loss(discriminate_x, discriminate_fake_x)
      Y_loss = self.discriminator_loss(discriminate_y, discriminate_fake_y)

      result.update({'loss_X/loss': X_loss, 'loss_Y/loss': Y_loss})

    self.G_optimizer.minimize(loss=G_total_loss,
                              var_list=self.G.trainable_variables,
                              tape=tape)
    self.F_optimizer.minimize(loss=F_total_loss,
                              var_list=self.F.trainable_variables,
                              tape=tape)
    self.X_optimizer.minimize(loss=X_loss,
                              var_list=self.X.trainable_variables,
                              tape=tape)
    self.Y_optimizer.minimize(loss=Y_loss,
                              var_list=self.Y.trainable_variables,
                              tape=tape)

    return result

  def reduce_dict(self, d: dict):
    """ reduce items in dictionary d """
    for k, v in d.items():
      d[k] = self.strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)

  @tf.function
  def distributed_train_step(self, x, y):
    results = self.strategy.run(self.train_step, args=(x, y))
    self.reduce_dict(results)
    return results

  def test_step(self, x, y):
    result = {}

    fake_x, fake_y, cycle_x, cycle_y = self.cycle_step(x, y, training=False)

    discriminate_fake_x = self.X(fake_x, training=False)
    discriminate_fake_y = self.Y(fake_y, training=False)

    G_loss = self.generator_loss(discriminate_fake_y)
    F_loss = self.generator_loss(discriminate_fake_x)

    F_cycle_loss = self.cycle_loss(x, cycle_x)
    G_cycle_loss = self.cycle_loss(y, cycle_y)

    same_x = self.F(x, training=False)
    same_y = self.G(y, training=False)
    G_identity_loss = self.identity_loss(y, same_y)
    F_identity_loss = self.identity_loss(x, same_x)

    G_total_loss = G_loss + G_cycle_loss + G_identity_loss
    F_total_loss = F_loss + F_cycle_loss + F_identity_loss

    result.update({
        'loss_G/loss': G_loss,
        'loss_G/cycle': G_cycle_loss,
        'loss_G/identity': G_identity_loss,
        'loss_G/total': G_total_loss,
        'loss_F/loss': F_loss,
        'loss_F/cycle': F_cycle_loss,
        'loss_F/identity': F_identity_loss,
        'loss_F/total': F_total_loss
    })

    discriminate_x = self.X(x, training=False)
    discriminate_y = self.Y(y, training=False)

    X_loss = self.discriminator_loss(discriminate_x, discriminate_fake_x)
    Y_loss = self.discriminator_loss(discriminate_y, discriminate_fake_y)

    result.update({
        'loss_X/loss': X_loss,
        'loss_Y/loss': Y_loss,
        'error/MAE(X, F(G(X)))': self.reduce_mean(MAE(x, cycle_x)),
        'error/MAE(Y, G(F(Y)))': self.reduce_mean(MAE(y, cycle_y)),
        'error/MAE(X, F(X))': self.reduce_mean(MAE(x, same_x)),
        'error/MAE(Y, G(Y))': self.reduce_mean(MAE(y, same_y))
    })

    return result

  @tf.function
  def distributed_test_step(self, x, y):
    results = self.strategy.run(self.test_step, args=(x, y))
    self.reduce_dict(results)
    return results


def train(args, train_ds, gan, summary, epoch: int):
  results = {}
  for x, y in tqdm(train_ds,
                   desc='Train',
                   total=args.train_steps,
                   disable=args.verbose == 0):
    result = gan.distributed_train_step(x, y)
    utils.append_dict(results, result)
  for key, value in results.items():
    summary.scalar(key, tf.reduce_mean(value), step=epoch, training=True)


def test(args, test_ds, gan, summary, epoch: int):
  results = {}
  for x, y in tqdm(test_ds,
                   desc='Test',
                   total=args.test_steps,
                   disable=args.verbose == 0):
    result = gan.distributed_test_step(x, y)
    utils.append_dict(results, result)
  for key, value in results.items():
    results[key] = tf.reduce_mean(value).numpy()
    summary.scalar(key, results[key], step=epoch, training=False)
  return results


def main(args):
  if args.clear_output_dir and os.path.exists(args.output_dir):
    rmtree(args.output_dir)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  tf.keras.backend.clear_session()

  np.random.seed(1234)
  tf.random.set_seed(1234)

  # initialize tf.distribute.MirroredStrategy
  strategy = tf.distribute.MirroredStrategy(devices=None)
  num_devices = strategy.num_replicas_in_sync
  args.global_batch_size = num_devices * args.batch_size
  print(f'Number of devices: {num_devices}')

  train_ds, test_ds, sample_ds = get_datasets(args, strategy=strategy)

  gan = CycleGAN(args, strategy=strategy)

  # initialize TensorBoard summary helper
  summary = utils.Summary(args.output_dir)

  epoch = gan.load_checkpoint()
  while (epoch := epoch + 1) < args.epochs:
    print(f'Epoch {epoch + 1:03d}/{args.epochs:03d}')

    start = time()
    train(args, train_ds, gan, summary, epoch)
    results = test(args, test_ds, gan, summary, epoch)
    end = time()
    summary.scalar('elapse', end - start, step=epoch)

    print(f'MAE(X, F(G(X))): {results["MAE(X, F(G(X)))"]:.04f}\t\t'
          f'MAE(X, F(X)): {results["MAE(Y, G(F(Y)))"]:.04f}\n'
          f'MAE(Y, G(F(Y))): {results["MAE(X, F(X))"]:.04f}\t\t'
          f'MAE(Y, G(Y)): {results["MAE(Y, G(Y))"]:.04f}\n'
          f'Elapse: {end - start:.02f}s\n')

    if epoch % 10 == 0 or epoch == args.epochs - 1:
      gan.save_checkpoint(epoch)
      utils.plot_cycle(sample_ds, gan, summary, epoch)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--epochs', default=200, type=int)
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--verbose', default=1, choices=[0, 1, 2])
  parser.add_argument('--clear_output_dir', action='store_true')

  main(parser.parse_args())
