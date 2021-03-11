
import os
import subprocess
import json
import datetime
import argparse

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

def mnist_dataset(batch_size):
    path = os.environ['DSDIR']+'/MNIST/mnist.npz'
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path)
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).repeat().shuffle(60000).batch(batch_size)
    return train_dataset

def build_cnn_model():
    model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=128, type =int,
                        help='batch size. it will be divided in mini-batch for each worker')
    parser.add_argument('-e','--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    
    hvd.init()
    
    # display info
    if hvd.rank() == 0:
        print(">>> Training on ", hvd.size() // hvd.local_size(), " nodes and ", hvd.size(), " processes")
    print("- Process {} corresponds to GPU {} of node {}".format(hvd.rank(), hvd.local_rank(), hvd.rank() // hvd.local_size()))
    
    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
    mnist_model = build_cnn_model()
    
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(0.001 * hvd.size())
    
    # ### Get data
    dataset = mnist_dataset(args.batch_size)
    
    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = mnist_model(images, training=True)
            loss_value = loss(labels, probs)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value


    # Horovod: adjust number of steps based on number of GPUs.
    start = datetime.datetime.now()
    for batch, (images, labels) in enumerate(dataset.take(args.epochs * 500 // hvd.size())):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 100 == 0 and hvd.rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))
            
    duration = datetime.datetime.now() - start

    if hvd.rank() == 0:
        print(' -- Trained in ' + str(duration) + ' -- ')

if __name__ == '__main__':

    main()
