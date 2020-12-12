import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1024
IMG_HEIGHT, IMG_WIDTH = 256, 256


def random_crop(image):
  cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image


def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image


def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(
      image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)
  # random mirroring
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_train_image(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


def preprocess_validation_image(image, label):
  image = normalize(image)
  return image


def prepare_dataset(hparams, ds, train=False):
  ds = ds.map(
      preprocess_train_image if train else preprocess_validation_image,
      num_parallel_calls=AUTOTUNE)
  ds = ds.cache()
  ds = ds.shuffle(BUFFER_SIZE)
  ds = ds.batch(hparams.batch_size)
  if train:
    ds = ds.prefetch(BUFFER_SIZE)
  return ds


def cardinality(ds):
  return tf.data.experimental.cardinality(ds)


def get_datasets(hparams):

  dataset, metadata = tfds.load(
      'cycle_gan/horse2zebra',
      data_dir='dataset',
      with_info=True,
      as_supervised=True)

  train_horses, train_zebras = dataset['trainA'], dataset['trainB']
  test_horses, test_zebras = dataset['testA'], dataset['testB']

  x_train = prepare_dataset(hparams, train_horses, train=True)
  y_train = prepare_dataset(hparams, train_zebras, train=True)
  x_validation = prepare_dataset(hparams, test_horses, train=False)
  y_validation = prepare_dataset(hparams, test_zebras, train=False)

  hparams.train_steps = int(min(cardinality(x_train), cardinality(y_train)))
  hparams.validation_steps = int(
      min(cardinality(x_validation), cardinality(y_validation)))
  hparams.image_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

  return x_train, x_validation, y_train, y_validation
