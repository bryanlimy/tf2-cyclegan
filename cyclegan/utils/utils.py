import os


def save_models(hparams, G, F, X, Y):
  if not hasattr(hparams, 'checkpoint_dir'):
    hparams.checkpoint_dir = os.path.join(hparams.output_dir, 'checkpoints')
    if not os.path.exists(hparams.checkpoint_dir):
      os.makedirs(hparams.checkpoint_dir)

  G.save_weights(os.path.join(hparams.checkpoint_dir, 'generator_g'))
  F.save_weights(os.path.join(hparams.checkpoint_dir, 'generator_f'))
  X.save_weights(os.path.join(hparams.checkpoint_dir, 'discriminator_x'))
  Y.save_weights(os.path.join(hparams.checkpoint_dir, 'discriminator_y'))
  print(f'checkpoints saved at {hparams.checkpoint_dir}')


def plot_transformation(x, y, gan, summary, epoch):
  fake_y = gan.G(x, training=False)
  cycled_x = gan.F(fake_y, training=False)

  fake_x = gan.F(y, training=False)
  cycled_y = gan.G(fake_x, training=False)

  # plot the first 3 transformations
  for i in range(3):
    summary.plot_transformation(
        f'X cycle/image_{i + 1:02d}',
        images=[
            x[i, ...],
            fake_y[i, ...],
            cycled_x[i, ...],
        ],
        labels=['X', 'G(X)', 'F(G(X))'],
        step=epoch,
        training=False)

    summary.plot_transformation(
        f'Y cycle/image_{i + 1:02d}',
        images=[
            y[i, ...],
            fake_x[i, ...],
            cycled_y[i, ...],
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
