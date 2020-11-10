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
  fake_x, fake_y, cycled_x, cycled_y = gan.cycle_step(x, y, training=False)
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
