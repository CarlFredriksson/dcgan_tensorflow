import numpy as np
import tensorflow as tf
import utils

def generator(Z, is_training):
    with tf.variable_scope("Generator"):
        x = Z
        # x.shape: (?, 1, 1, 100)

        x = tf.layers.conv2d_transpose(x, 256, 7, strides=1, padding="valid", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 7, 7, 256)

        x = tf.layers.conv2d_transpose(x, 128, 5, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 14, 14, 128)

        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding="same")
        x = tf.nn.tanh(x)
        # x.shape: (?, 28, 28, 1)

    return x

def discriminator(X, is_training, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        x = X
        # x.shape: (?, 28, 28, 1)

        x = tf.layers.conv2d(x, 128, 5, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 14, 14, 128)

        x = tf.layers.conv2d(x, 256, 5, strides=2, padding="same", use_bias=False)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # x.shape: (?, 7, 7, 256)

        x = tf.layers.conv2d(x, 1, 7, strides=1, padding="valid")
        x = tf.nn.sigmoid(x)
        # x.shape: (?, 1, 1, 1)

    return x

def create_gan(X, Z, is_training):
    G = generator(Z, is_training)
    D_real = discriminator(X, is_training)
    D_fake = discriminator(G, is_training, reuse=True)
    
    return G, D_real, D_fake

# Define constants
NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA1 = 0.5
NOISE_DIM = 100
SAMPLE_SIZE = 100

# Load mnist data
X_train = utils.load_mnist_data()
utils.plot_sample(X_train[:SAMPLE_SIZE], "output/mnist_data.png")
X_train = utils.preprocess_images(X_train)
mini_batches = utils.random_mini_batches(X_train, BATCH_SIZE)

# Create DCGAN
X = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
Z = tf.placeholder(tf.float32, [None, 1, 1, NOISE_DIM])
is_training = tf.placeholder(tf.bool, shape=())
G, D_real, D_fake = create_gan(X, Z, is_training)

# Create training steps
G_loss_func, D_loss_func = utils.create_loss_funcs(D_real, D_fake)
G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Generator")
D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Discriminator")
G_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1).minimize(G_loss_func, var_list=G_vars)
D_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=BETA1).minimize(D_loss_func, var_list=D_vars)

# Start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for X_batch in mini_batches:
            Z_batch = utils.generate_Z_batch((X_batch.shape[0], 1, 1, NOISE_DIM))

            # Compute losses
            G_loss, D_loss = sess.run([G_loss_func, D_loss_func], feed_dict={X: X_batch, Z: Z_batch, is_training: True})
            print("Epoch [{0}/{1}] - G_loss: {2}, D_loss: {3}".format(epoch, NUM_EPOCHS - 1, G_loss, D_loss))

            # Run training steps
            _ = sess.run(G_train_step, feed_dict={Z: Z_batch, is_training: True})
            _ = sess.run(D_train_step, feed_dict={X: X_batch, Z: Z_batch, is_training: True})

        # Plot generated images
        Z_batch = utils.generate_Z_batch((SAMPLE_SIZE, 1, 1, NOISE_DIM))
        gen_imgs = sess.run(G, feed_dict={Z: Z_batch, is_training: True})
        gen_imgs = utils.postprocess_images(gen_imgs)
        utils.plot_sample(gen_imgs, "output/dcgan/dcgan_gen_data_" + str(epoch) + ".png")
