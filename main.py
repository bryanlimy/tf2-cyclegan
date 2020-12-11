import os
import argparse
import numpy as np
from tqdm import tqdm
from time import time
import tensorflow as tf
from shutil import rmtree
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from cyclegan.utils import utils
from cyclegan.models.registry import get_models
from cyclegan.utils.summary_helper import Summary
from cyclegan.alogrithms.registry import get_algorithm
from cyclegan.utils.dataset_helper import get_datasets


def set_precision_policy(hparams):
  policy = mixed_precision.Policy('mixed_float16' if hparams.
                                  mixed_precision else 'float32')
  mixed_precision.set_policy(policy)
  if hparams.verbose:
    print(f'\nCompute dtype: {policy.compute_dtype}\n'
          f'Variable dtype: {policy.variable_dtype}\n')
  return policy


def train(hparams, x_ds, y_ds, gan, summary, epoch):
  metrics = {}
  for x, y in tqdm(
      tf.data.Dataset.zip((x_ds, y_ds)),
      desc='Train',
      total=hparams.train_steps):
    result = gan.train(x, y)
    hparams.global_step += 1
    utils.update_dict(metrics, result, replace=False)
  for key, value in metrics.items():
    summary.scalar(f'loss/{key}', tf.reduce_mean(value), epoch, training=True)


def validate(hparams, x_ds, y_ds, gan, summary, epoch):
  metrics = {}
  for x, y in tqdm(
      tf.data.Dataset.zip((x_ds, y_ds)),
      desc='Validation',
      total=hparams.validation_steps):
    result = gan.validate(x, y)
    utils.update_dict(metrics, result, replace=False)
  metrics = {key: np.mean(value) for key, value in metrics.items()}
  for key, value in metrics.items():
    summary.scalar(f'mse/{key}', value, epoch, training=False)
  return metrics


def main(hparams):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  tf.keras.backend.clear_session()

  set_precision_policy(hparams)

  summary = Summary(hparams)
  x_train, x_validation, y_train, y_validation = get_datasets(hparams)
  G, Y = get_models(hparams, summary, name='G')
  F, X = get_models(hparams, summary, name='F')
  gan = get_algorithm(hparams, G, F, X, Y)

  # store the images to plot
  plot_images = next(iter(tf.data.Dataset.zip((x_validation, y_validation))))

  for epoch in range(hparams.epochs):
    print('Epoch {:03d}/{:03d}'.format(epoch + 1, hparams.epochs))

    start = time()
    train(hparams, x_train, y_train, gan, summary, epoch)
    metrics = validate(hparams, x_validation, y_validation, gan, summary, epoch)
    end = time()
    summary.scalar('elapse', end - start, step=epoch)

    print(f'MSE(X, F(G(X))): {metrics["MSE(X, F(G(X)))"]:.04f}\t\t'
          f'MSE(X, F(X)): {metrics["MSE(Y, G(F(Y)))"]:.04f}\n'
          f'MSE(Y, G(F(Y))): {metrics["MSE(X, F(X))"]:.04f}\t\t'
          f'MSE(Y, G(Y)): {metrics["MSE(Y, G(Y))"]:.04f}\n'
          f'Elapse: {(end - start) / 60:.02f} mins\n')

    if epoch % 10 == 0 or epoch == hparams.epochs - 1:
      utils.plot_transformation(plot_images[0], plot_images[1], gan, summary,
                                epoch)

  utils.save_models(hparams, G, F, X, Y)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--algorithm', default='gan', type=str)
  parser.add_argument('--model', default='unet', type=str)
  parser.add_argument('--activation', default='elu', type=str)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument(
      '--normalizer',
      default='layer_norm',
      choices=[None, 'layer_norm', 'batch_norm', 'instance_norm'])
  parser.add_argument('--initializer', default='glorot_uniform', type=str)
  parser.add_argument(
      '--alpha', default=10., type=float, help='cycle loss coefficient')
  parser.add_argument(
      '--beta', default=5., type=float, help='identity loss coefficient')
  parser.add_argument('--epochs', default=200, type=int)
  parser.add_argument('--batch_size', default=32, type=int)
  parser.add_argument('--learning_rate', default=2e-4, type=float)
  parser.add_argument('--critic_steps', default=5, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument(
      '--error',
      default='mae',
      choices=['mse', 'mae', 'huber'],
      help='error function to measure cycle loss and identity loss')
  parser.add_argument('--mixed_precision', action='store_true')
  parser.add_argument('--verbose', default=1, type=int)
  parser.add_argument('--clear_output_dir', action='store_true')
  params = parser.parse_args()

  np.random.seed(1234)
  tf.random.set_seed(1234)

  params.global_step = 0
  main(params)
