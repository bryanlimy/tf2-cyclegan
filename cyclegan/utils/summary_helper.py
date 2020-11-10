import os
import io
import platform
import tensorflow as tf

import matplotlib
if platform.system() == 'Darwin':
  matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


class Summary(object):
  """ 
  Log tf.Summary to output_dir during training and output_dir/eval during 
  evaluation
  """

  def __init__(self, hparams):
    self._hparams = hparams

    self.train_writer = tf.summary.create_file_writer(hparams.output_dir)
    self.val_writer = tf.summary.create_file_writer(
        os.path.join(os.path.join(hparams.output_dir, 'validation')))

    tick_size = 12
    legend_size = 12
    label_size = 14
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.rc('axes', titlesize=label_size)
    plt.rc('axes', labelsize=label_size)
    plt.rc('legend', fontsize=legend_size)

    self.dpi = 100
    self.framerate = 24

  def _get_writer(self, training):
    return self.train_writer if training else self.val_writer

  def _plot_to_png(self):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=self.dpi, format='png')
    buffer.seek(0)
    return tf.image.decode_png(buffer.getvalue(), channels=4)

  def scalar(self, tag, value, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.scalar(tag, value, step=step)

  def histogram(self, tag, values, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.histogram(tag, values, step=step)

  def image(self, tag, values, step=0, training=True):
    writer = self._get_writer(training)
    with writer.as_default():
      tf.summary.image(tag, data=values, step=step, max_outputs=len(values))

  def plot_transformation(self,
                          tag,
                          images,
                          labels,
                          title=None,
                          step=0,
                          training=True):
    assert type(images) == type(labels) == list
    assert len(images) == len(labels)

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('white')

    for i in range(len(images)):
      plt.subplot(1, len(images), i + 1)
      plt.imshow(images[i] * 0.5 + 0.5)
      plt.title(labels[i])
      axis = plt.gca()
      axis.set_xticks([])
      axis.set_yticks([])
      plt.tight_layout()

    fig.tight_layout()
    image = self._plot_to_png()
    plt.close()
    self.image(tag, values=[image], step=step, training=training)
