import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (X_train, _), _ = mnist.load_data()
    return X_train

def preprocess_images(X):
    X = X / 255
    X = X - 0.5
    X = X * 2
    X = np.expand_dims(X, axis=-1)
    return X

def postprocess_images(X):
    X = np.squeeze(X, axis=-1)
    X = X / 2
    X = X + 0.5
    X = X * 255
    return X

def generate_Z_batch(size):
    return np.random.uniform(low=-1, high=1, size=size)

def plot_sample(sample, path):
    n = int(np.sqrt(sample.shape[0]))
    fig = plt.figure(figsize=(8, 8))
    for i in range(n*n):
        ax = plt.subplot(n, n, i + 1)
        ax.imshow(sample[i], cmap=plt.get_cmap("gray"))
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=0.025, wspace=0.025)
    plt.savefig(path, bbox_inches="tight")
    plt.clf()
    plt.close()

def random_mini_batches(X, batch_size):
    mini_batches = []
    m = X.shape[0]
    np.random.shuffle(X)

    # Partition into mini-batches
    num_complete_batches = math.floor(m / batch_size)
    for i in range(num_complete_batches):
        batch = X[i * batch_size : (i + 1) * batch_size]
        mini_batches.append(batch)

    # Handling the case that the last mini-batch < batch_size
    if m % batch_size != 0:
        batch = X[num_complete_batches * batch_size : m]
        mini_batches.append(batch)

    return mini_batches

def create_loss_funcs(D_real, D_fake):
    eps = 1e-12
    G_loss_func = tf.reduce_mean(-tf.log(D_fake + eps))
    D_loss_func = tf.reduce_mean(-(tf.log(D_real + eps) + tf.log(1 - D_fake + eps)))

    return G_loss_func, D_loss_func
