## TensorFlow 2 implementation of CycleGAN with WGAN-GP
This repository implements CycleGAN ([Zhu et. al. 2017](https://arxiv.org/pdf/1703.10593.pdf)) using TensorFlow 2 with code modularity in mind. The following is a list of features that this codebase introduces:

- WGAN-GP ([Gulrajani et. al. 2017](https://arxiv.org/pdf/1704.00028.pdf)) formulation using the flag `--algorithm wgan_gp` .
- [mixed-precision](https://www.tensorflow.org/guide/mixed_precision) training using the flag `--mixed_precision`.
- modularity
    - define different model architectures under `cyclegan/models` and use them by `--model unet`.
    - define different objective functions under `cyclegan/algorithms` and use them by `--algorithm gan`.
    - different error function to calculate cycle loss and identity loss with `--cycle_error mse`.
- `tf.summary` to log and monitor model performance.

### 1. Installation
- create virtual environment for the project
  ```
  conda create -n cyclegan python=3.6
  ```
- activate virtual environment
  ```
  conda activate cyclegan
  ```
- install required packages
  ```
  sh setup.sh
  ```

### 2. Run codebase
- Use `--help` to see all available flags.
- By default, we train our model using the `horse2zebra` dataset from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cycle_gan#cycle_ganhorse2zebra).
- The training logs and checkpoints are stored in `--output_dir`
- To train our model in mixed-precision with the UNet architecture and WGAN-GP formulation, we can use the following command
  ```
  python main.py --output_dir runs/001_unet_wgangp --alogrithm unet --model unet --mixed_precision
  ``` 
- monitor training performance using `tensorboard`
  ```
  tensorboard --logdir runs/001_unet_wgangp
  ```

   