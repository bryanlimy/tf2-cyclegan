import os
import io
import typing as t
import numpy as np
from math import ceil
from tqdm import tqdm
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras import layers

BATCH_SIZE = 2  # per replica batch size

# initialize tf.distribute.MirroredStrategy
strategy = tf.distribute.MirroredStrategy(devices=None)
GLOBAL_BATCH_SIZE = strategy.num_replicas_in_sync * BATCH_SIZE

print(f'Number of devices: {strategy.num_replicas_in_sync}')

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SHAPE = (286, 286)
INPUT_SHAPE = (256, 256, 3)

dataset, metadata = tfds.load("cycle_gan/horse2zebra",
                              with_info=True,
                              as_supervised=True)
train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
test_horses, test_zebras = dataset["testA"], dataset["testB"]

# calculate the number of train and test steps needed per epoch
get_size = lambda name: metadata.splits.get(name).num_examples
NUM_TRAIN_SAMPLES = min([get_size('trainA'), get_size('trainB')])
NUM_TEST_SAMPLES = min([get_size('testA'), get_size('testB')])
TRAIN_STEPS = ceil(NUM_TRAIN_SAMPLES / GLOBAL_BATCH_SIZE)
TEST_STEPS = ceil(NUM_TEST_SAMPLES / GLOBAL_BATCH_SIZE)


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


train_horses = train_horses.take(NUM_TRAIN_SAMPLES)
train_horses = train_horses.map(preprocess_train, num_parallel_calls=AUTOTUNE)
train_horses = train_horses.cache()
train_horses = train_horses.shuffle(buffer_size=256)

train_zebras = train_zebras.take(NUM_TRAIN_SAMPLES)
train_zebras = train_zebras.map(preprocess_train, num_parallel_calls=AUTOTUNE)
train_zebras = train_zebras.cache()
train_zebras = train_zebras.shuffle(buffer_size=256)

# create train dataset
train_ds = tf.data.Dataset.zip((train_horses.batch(GLOBAL_BATCH_SIZE),
                                train_zebras.batch(GLOBAL_BATCH_SIZE)))
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

test_horses = test_horses.take(NUM_TEST_SAMPLES)
test_horses = test_horses.map(preprocess_test, num_parallel_calls=AUTOTUNE)
test_horses = test_horses.cache()

test_zebras = test_zebras.take(NUM_TEST_SAMPLES)
test_zebras = test_zebras.map(preprocess_test, num_parallel_calls=AUTOTUNE)
test_zebras = test_zebras.cache()

# create test dataset
test_ds = tf.data.Dataset.zip((test_horses.batch(GLOBAL_BATCH_SIZE),
                               test_zebras.batch(GLOBAL_BATCH_SIZE)))

# take 5 samples from the test set for plotting
plot_ds = tf.data.Dataset.zip(
    (test_horses.take(5).batch(1), test_zebras.take(5).batch(1)))

# create distributed datasets
train_ds = strategy.experimental_distribute_dataset(train_ds)
test_ds = strategy.experimental_distribute_dataset(test_ds)


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


kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


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
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
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
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
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
                  name: str = 'generator'):
  inputs = layers.Input(shape=input_shape, name='inputs')

  outputs = ReflectionPadding2D(paddings=(3, 3))(inputs)
  outputs = layers.Conv2D(filters=filters, kernel_size=(7, 7),
                          use_bias=False)(outputs)
  outputs = tfa.layers.InstanceNormalization(
      gamma_initializer=gamma_initializer)(outputs)
  outputs = layers.Activation("relu")(outputs)

  # down sample
  for _ in range(num_downsampling_blocks):
    filters *= 2
    outputs = downsample(outputs,
                         filters=filters,
                         activation=layers.Activation("relu"))

  # residual blocks
  for _ in range(num_residual_blocks):
    outputs = residual_block(outputs, activation=layers.Activation("relu"))

  # up sample
  for _ in range(num_upsample_blocks):
    filters //= 2
    outputs = upsample(outputs, filters, activation=layers.Activation("relu"))

  # output block
  outputs = ReflectionPadding2D(paddings=(3, 3))(outputs)
  outputs = layers.Conv2D(filters=3, kernel_size=(7, 7),
                          padding="valid")(outputs)
  outputs = layers.Activation("tanh")(outputs)

  return tf.keras.models.Model(inputs, outputs, name=name)


def get_discriminator(input_shape: tuple,
                      filters: int = 64,
                      num_downsampling: int = 3,
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


OUTPUT_DIR = 'runs'  # directory to store checkpoint and TensorBoard summary

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

with strategy.scope():
  # initialize models
  gen_G = get_generator(input_shape=INPUT_SHAPE, name='gen_G')
  gen_F = get_generator(input_shape=INPUT_SHAPE, name='gen_F')
  dis_X = get_discriminator(input_shape=INPUT_SHAPE, name='dis_X')
  dis_Y = get_discriminator(input_shape=INPUT_SHAPE, name='dis_Y')

  # initialize optimizers
  G_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                         beta_1=0.5,
                                         beta_2=0.9)
  F_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                         beta_1=0.5,
                                         beta_2=0.9)
  X_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                         beta_1=0.5,
                                         beta_2=0.9)
  Y_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                         beta_1=0.5,
                                         beta_2=0.9)

  # initialize checkpoint
  checkpoint = tf.train.Checkpoint(gen_G=gen_G,
                                   gen_F=gen_F,
                                   dis_X=dis_X,
                                   dis_Y=dis_Y,
                                   G_optimizer=G_optimizer,
                                   F_optimizer=F_optimizer,
                                   X_optimizer=X_optimizer,
                                   Y_optimizer=Y_optimizer)


class Summary:
  """ Helper class to write TensorBoard summaries """

  def __init__(self, output_dir: str):
    self.dpi = 120
    plt.style.use('seaborn-deep')

    self.writers = [
        tf.summary.create_file_writer(output_dir),
        tf.summary.create_file_writer(os.path.join(output_dir, 'test'))
    ]

  def get_writer(self, training: bool):
    return self.writers[0 if training else 1]

  def scalar(self, tag, value, step: int = 0, training: bool = False):
    writer = self.get_writer(training)
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def image(self, tag, values, step: int = 0, training: bool = False):
    writer = self.get_writer(training)
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=len(values))

  def figure(self,
             tag,
             figure,
             step: int = 0,
             training: bool = False,
             close: bool = True):
    """ Write matplotlib figure to summary
    Args:
      tag: data identifier
      figure: matplotlib figure or a list of figures
      step: global step value to record
      training: training summary or test summary
      close: flag to close figure
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, dpi=self.dpi, format='png', bbox_inches='tight')
    buffer.seek(0)
    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    self.image(tag, tf.expand_dims(image, 0), step=step, training=training)
    if close:
      plt.close(figure)

  def image_cycle(self,
                  tag: str,
                  images: t.List[np.ndarray],
                  labels: t.List[str],
                  step: int = 0,
                  training: bool = False):
    """ Plot image cycle to TensorBoard
    Args:
      tags: data identifier
      images: list of np.ndarray where len(images) == 3 and each array has
              shape (N,H,W,C)
      labels: list of string where len(labels) == 3
      step: global step value to record
      training: training summary or test summary
    """
    assert len(images) == len(labels) == 3
    for sample in range(len(images[0])):
      figure, axes = plt.subplots(nrows=1,
                                  ncols=3,
                                  figsize=(9, 3.25),
                                  dpi=self.dpi)
      axes[0].imshow(images[0][sample, ...], interpolation='none')
      axes[0].set_title(labels[0])

      axes[1].imshow(images[1][sample, ...], interpolation='none')
      axes[1].set_title(labels[1])

      axes[2].imshow(images[2][sample, ...], interpolation='none')
      axes[2].set_title(labels[2])

      plt.setp(axes, xticks=[], yticks=[])
      plt.tight_layout()
      figure.subplots_adjust(wspace=0.02, hspace=0.02)
      self.figure(tag=f'{tag}/sample_#{sample:03d}',
                  figure=figure,
                  step=step,
                  training=training,
                  close=True)


def append_dict(dict1: dict, dict2: dict, replace: bool = False):
  """ append items in dict2 to dict1 """
  for key, value in dict2.items():
    if replace:
      dict1[key] = value
    else:
      if key not in dict1:
        dict1[key] = []
      dict1[key].append(value)


def plot_cycle(ds, summary, epoch: int):
  """ plot X -> G(X) -> F(G(X)) and Y -> F(Y) -> G(F(Y)) """
  samples = {}
  for x, y in ds:
    fake_y = gen_G(x, training=False)
    cycle_x = gen_F(fake_y, training=False)
    fake_x = gen_F(y, training=False)
    cycle_y = gen_G(fake_x, training=False)
    append_dict(dict1=samples,
                dict2={
                    'x': x,
                    'y': y,
                    'fake_x': fake_x,
                    'fake_y': fake_y,
                    'cycle_x': cycle_x,
                    'cycle_y': cycle_y
                })
  for key, images in samples.items():
    # scale images back to [0, 255]
    images = tf.concat(images, axis=0).numpy()
    images = ((images + 1) * 127.5).astype(np.uint8)
    samples[key] = images
  summary.image_cycle(
      tag=f'X_cycle',
      images=[samples['x'], samples['fake_y'], samples['cycle_x']],
      labels=['X', 'G(X)', 'F(G(X))'],
      step=epoch,
      training=False)
  summary.image_cycle(
      tag=f'Y_cycle',
      images=[samples['y'], samples['fake_x'], samples['cycle_y']],
      labels=['Y', 'F(Y)', 'G(F(Y))'],
      step=epoch,
      training=False)


summary = Summary(output_dir=OUTPUT_DIR)

LAMBDA_CYCLE = 10.0  # cycle consistent loss coefficient
LAMBDA_IDENTITY = 0.5 * LAMBDA_CYCLE  # identity loss coefficient


def reduce_mean(per_sample_loss):
  """ return the global mean of per-sample loss """
  return tf.reduce_sum(per_sample_loss) / GLOBAL_BATCH_SIZE


def generator_loss(discriminate_fake):
  per_sample_loss = MSE(y_true=tf.ones_like(discriminate_fake),
                        y_pred=discriminate_fake)
  return reduce_mean(per_sample_loss)


def cycle_loss(real_samples, cycle_samples):
  per_sample_loss = MAE(y_true=real_samples, y_pred=cycle_samples)
  return LAMBDA_CYCLE * reduce_mean(per_sample_loss)


def identity_loss(real_samples, identity_samples):
  per_sample_loss = MAE(y_true=real_samples, y_pred=identity_samples)
  return LAMBDA_IDENTITY * reduce_mean(per_sample_loss)


def discriminator_loss(discriminate_real, discriminate_fake):
  real_loss = MSE(y_true=tf.ones_like(discriminate_real),
                  y_pred=discriminate_real)
  fake_loss = MSE(y_true=tf.zeros_like(discriminate_fake),
                  y_pred=discriminate_fake)
  per_sample_loss = 0.5 * (real_loss + fake_loss)
  return reduce_mean(per_sample_loss)


def train_step(x, y):
  result = {}

  with tf.GradientTape(persistent=True) as tape:
    fake_y = gen_G(x, training=True)
    fake_x = gen_F(y, training=True)

    discriminate_fake_x = dis_X(fake_x, training=True)
    discriminate_fake_y = dis_Y(fake_y, training=True)

    G_loss = generator_loss(discriminate_fake_y)
    F_loss = generator_loss(discriminate_fake_x)

    G_cycle_loss = cycle_loss(y, gen_G(fake_x, training=True))
    F_cycle_loss = cycle_loss(x, gen_F(fake_y, training=True))

    G_identity_loss = identity_loss(y, gen_G(y, training=True))
    F_identity_loss = identity_loss(x, gen_F(x, training=True))

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

    discriminate_x = dis_X(x, training=True)
    discriminate_y = dis_Y(y, training=True)
    discriminate_fake_x = dis_X(fake_x, training=True)
    discriminate_fake_y = dis_Y(fake_y, training=True)

    X_loss = discriminator_loss(discriminate_x, discriminate_fake_x)
    Y_loss = discriminator_loss(discriminate_y, discriminate_fake_y)

    result.update({'loss_X/loss': X_loss, 'loss_Y/loss': Y_loss})

  G_optimizer.minimize(loss=G_total_loss,
                       var_list=gen_G.trainable_variables,
                       tape=tape)
  F_optimizer.minimize(loss=F_total_loss,
                       var_list=gen_F.trainable_variables,
                       tape=tape)
  X_optimizer.minimize(loss=X_loss,
                       var_list=dis_X.trainable_variables,
                       tape=tape)
  Y_optimizer.minimize(loss=Y_loss,
                       var_list=dis_Y.trainable_variables,
                       tape=tape)

  return result


def reduce_dict(d: dict):
  """ inplace reduction of items in dictionary d """
  return {
      k: strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
      for k, v in d.items()
  }


@tf.function
def distributed_train_step(x, y):
  results = strategy.run(train_step, args=(x, y))
  results = reduce_dict(results)
  return results


def test_step(x, y):
  result = {}

  fake_y = gen_G(x, training=False)
  cycle_x = gen_F(fake_y, training=False)

  fake_x = gen_F(y, training=False)
  cycle_y = gen_G(fake_x, training=False)

  discriminate_fake_x = dis_X(fake_x, training=False)
  discriminate_fake_y = dis_Y(fake_y, training=False)

  G_loss = generator_loss(discriminate_fake_y)
  F_loss = generator_loss(discriminate_fake_x)

  F_cycle_loss = cycle_loss(x, cycle_x)
  G_cycle_loss = cycle_loss(y, cycle_y)

  same_x = gen_F(x, training=False)
  same_y = gen_G(y, training=False)
  G_identity_loss = identity_loss(y, same_y)
  F_identity_loss = identity_loss(x, same_x)

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

  discriminate_x = dis_X(x, training=False)
  discriminate_y = dis_Y(y, training=False)

  X_loss = discriminator_loss(discriminate_x, discriminate_fake_x)
  Y_loss = discriminator_loss(discriminate_y, discriminate_fake_y)

  result.update({
      'loss_X/loss': X_loss,
      'loss_Y/loss': Y_loss,
      'error/MAE(X, F(G(X)))': reduce_mean(MAE(x, cycle_x)),
      'error/MAE(Y, G(F(Y)))': reduce_mean(MAE(y, cycle_y)),
      'error/MAE(X, F(X))': reduce_mean(MAE(x, same_x)),
      'error/MAE(Y, G(Y))': reduce_mean(MAE(y, same_y))
  })

  return result


@tf.function
def distributed_test_step(x, y):
  results = strategy.run(test_step, args=(x, y))
  results = reduce_dict(results)
  return results


def train(ds, summary, epoch: int):
  results = {}
  for x, y in tqdm(ds, desc='Train', total=TRAIN_STEPS):
    result = distributed_train_step(x, y)
    append_dict(results, result)
  for key, value in results.items():
    summary.scalar(key, tf.reduce_mean(value), step=epoch, training=True)


def test(ds, summary, epoch: int):
  results = {}
  for x, y in tqdm(ds, desc='Test', total=TEST_STEPS):
    result = distributed_test_step(x, y)
    append_dict(results, result)
  for key, value in results.items():
    results[key] = tf.reduce_mean(value).numpy()
    summary.scalar(key, results[key], step=epoch, training=False)
  return results


NUM_EPOCHS = 200

for epoch in range(NUM_EPOCHS):
  print(f'Epoch {epoch + 1:03d}/{NUM_EPOCHS:03d}')

  start = time()
  train(train_ds, summary, epoch)
  results = test(test_ds, summary, epoch)
  end = time()

  print(f'MAE(X, F(G(X))): {results["error/MAE(X, F(G(X)))"]:.04f}\t\t'
        f'MAE(X, F(X)): {results["error/MAE(Y, G(F(Y)))"]:.04f}\n'
        f'MAE(Y, G(F(Y))): {results["error/MAE(X, F(X))"]:.04f}\t\t'
        f'MAE(Y, G(Y)): {results["error/MAE(Y, G(Y))"]:.04f}\n'
        f'Elapse: {end - start:.02f}s\n')

  if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
    checkpoint.write(os.path.join(OUTPUT_DIR, 'checkpoint'))
    plot_cycle(plot_ds, summary, epoch)
