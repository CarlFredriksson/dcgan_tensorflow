import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.keras.datasets import mnist

EPS = 1e-12
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA1 = 0.5
NOISE_DIM = 100
IMG_DIMS = (28, 28)

def load_mnist_data():
    (X_train, _), _ = mnist.load_data()

    return X_train

def preprocess(data):
    data = data / 255
    data = data - 0.5
    data = data * 2
    data = np.expand_dims(data, axis=-1)
    return data

def postprocess(data):
    data = np.squeeze(data, axis=-1)
    data = data / 2
    data = data + 0.5
    data = data * 255
    return data

def plot_data(mnist_data, plot_name):
    n = 7
    fig = plt.figure(figsize=(10, 10))
    for i in range(n*n):
        ax = plt.subplot(n, n, i + 1)
        ax.imshow(mnist_data[i], cmap=plt.get_cmap("gray"))
        ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=0.025, wspace=0.025)
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def random_mini_batches(data, batch_size):
    mini_batches = []
    m = data.shape[0]
    np.random.shuffle(data)

    # Partition into mini-batches
    num_complete_batches = math.floor(m / batch_size)
    for i in range(num_complete_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        mini_batches.append(batch)

    # Handling the case that the last mini-batch < batch_size
    if m % batch_size != 0:
        batch = data[num_complete_batches * batch_size : m]
        mini_batches.append(batch)

    return mini_batches

def generate_Z_batch(size):
    return np.random.uniform(low=-1, high=1, size=size)

def generator(Z, is_training):
    with tf.variable_scope("GAN/Generator"):
        x = Z
        # x.shape: (?, 100)

        # You don't need to add a bias term before applying batch norm.
        # Batch norm adds its own bias term.
        x = tf.layers.dense(x, units=7*7*256, use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 256])
        # x.shape: (?, 7, 7, 256)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=1, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 7, 7, 128)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 14, 14, 64)

        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=2, padding="same", use_bias=False)
        x = tf.nn.tanh(x)
        # x.shape: (?, 28, 28, 1)

    return x

def discriminator(X, is_training, reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        x = X
        # x.shape: (?, 28, 28, 1)

        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 14, 14, 64)

        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 7, 7, 128)

        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 1)
        x = tf.nn.sigmoid(x)
        # x.shape: (?, 1)

    return x

def create_model(X, Z, is_training):
    gen_out = generator(Z, is_training)
    disc_out_real = discriminator(X, is_training)
    disc_out_fake = discriminator(gen_out, is_training, reuse=True)

    return gen_out, disc_out_real, disc_out_fake

# Load mnist data
mnist_data = load_mnist_data()
plot_data(mnist_data, "mnist_data.png")
mnist_data = preprocess(mnist_data)
mini_batches = random_mini_batches(mnist_data, BATCH_SIZE)

# Create model
X = tf.placeholder(tf.float32, [None, IMG_DIMS[0], IMG_DIMS[1], 1])
Z = tf.placeholder(tf.float32, [None, NOISE_DIM])
is_training = tf.placeholder_with_default(False, shape=())
gen_out, disc_out_real, disc_out_fake = create_model(X, Z, is_training)

# Create loss functions
disc_loss_func = tf.reduce_mean(-(tf.log(disc_out_real + EPS) + tf.log(1 - disc_out_fake + EPS)))
gen_loss_func = tf.reduce_mean(-tf.log(disc_out_fake + EPS))

# Get variables
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

# Create training steps
gen_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1).minimize(gen_loss_func, var_list=gen_vars)
disc_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1).minimize(disc_loss_func, var_list=disc_vars)

# Start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for X_batch in mini_batches:
            Z_batch = generate_Z_batch((X_batch.shape[0], NOISE_DIM))

            # Compute losses
            gen_loss, disc_loss = sess.run([gen_loss_func, disc_loss_func], feed_dict={X: X_batch, Z: Z_batch, is_training: True})
            print("Epoch " + str(epoch) + " - gen_loss: " + str(gen_loss) + ", disc_loss: " + str(disc_loss))

            # Run training steps
            _ = sess.run(gen_train_step, feed_dict={Z: Z_batch, is_training: True})
            _ = sess.run(disc_train_step, feed_dict={X: X_batch, Z: Z_batch, is_training: True})

        # Plot generated images
        Z_batch = generate_Z_batch((BATCH_SIZE, NOISE_DIM))
        gen_imgs = sess.run(gen_out, feed_dict={Z: Z_batch, is_training: True})
        gen_imgs = postprocess(gen_imgs)
        plot_data(gen_imgs, "gen_data_" + str(epoch) + ".png")
