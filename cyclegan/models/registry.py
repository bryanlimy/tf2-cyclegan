import os
from .utils import count_trainable_params, model_summary

_MODELS = dict()


def register(name):

  def add_to_dict(fn):
    global _MODELS
    _MODELS[name] = fn
    return fn

  return add_to_dict


def get_models(hparams, summary, name=''):
  if hparams.model not in _MODELS:
    print('model {} not found'.format(hparams.model))
    exit()

  generator, discriminator = _MODELS[hparams.model](hparams, name=name)

  summary.scalar(f'model/trainable_parameters/generator_{name}',
                 count_trainable_params(generator))
  summary.scalar(f'model/trainable_parameters/discriminator_{name}',
                 count_trainable_params(discriminator))

  return generator, discriminator
