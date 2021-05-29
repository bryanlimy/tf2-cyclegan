# Distributed training with custom training loops in TensorFlow 2
This repository provide a concise example on how to use `tf.distribute.Strategy` with custom training loops in TensorFlow 2. We adapt the CycleGAN ([Zhu et. al. 2017](https://arxiv.org/pdf/1703.10593.pdf)) tutorials from [Keras](https://keras.io/examples/generative/cyclegan) and [TensorFlow](https://www.tensorflow.org/tutorials/generative/cyclegan) and train the model with multiple GPUs. See [medium.com](https://medium.com) for a detailed tutorial.

## 1. Setup
- create virtual environment for the project
  ```
  conda create -n cyclegan python=3.8
  ```
- activate virtual environment
  ```
  conda activate cyclegan
  ```
- install required packages
  ```
  sh setup.sh
  ```

## 2. Run
- Use `--help` to see all available flags.
- By default, we train our model using the `horse2zebra` dataset from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cycle_gan#cycle_ganhorse2zebra).
- Training logs and checkpoints are stored in `--output_dir`
- We can use the following command to train the CycleGAN model on 2 GPUs and store the TensorBoard summary and checkpoints to `runs/cyclegan`:
  ```
  CUDA_VISIBLE_DEVICES=1,2 python main.py --output_dir runs/cyclegan --epochs 200
  ``` 


## 3. Results
- Use `TensorBoard` to inspect the training summary and plots
  ```
  tensorboard --logdir runs/cyclegan
  ```