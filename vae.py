import numpy as np
import tensorflow as tf
import sys



winit = tf.contrib.layers.xavier_initializer() 


class VAE():

    def __init__(self, sess, lr=1e-4, batch_size=32, n_z=32, nr=82, nc=96, img_size=64, scope='vae'):
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.n_z = n_z
        self.nr = nr
        self.nc = nc  
        self.img_size = img_size
        self.scope = scope

        self.network()


    def network(self):

      with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.nr, self.nc, 3])
        self.beta = tf.placeholder(name='beta', dtype=tf.float32, shape=())
        self.is_train = tf.placeholder(name='is_train', dtype=tf.bool)

        # encoder 
        input_x = tf.image.resize_images(self.x, size=(64,64)) 
        self.input_x = input_x/255.0 
        conv1 = tf.layers.conv2d(self.input_x, filters=32, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='enc_conv1')
        conv1 = tf.layers.batch_normalization(conv1, training=self.is_train)
        conv1 = tf.nn.relu(conv1)

        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='enc_conv2')
        conv2 = tf.layers.batch_normalization(conv2, training=self.is_train)
        conv2 = tf.nn.relu(conv2)

        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='enc_conv3')
        conv3 = tf.layers.batch_normalization(conv3, training=self.is_train)
        conv3 = tf.nn.relu(conv3)

        conv4 = tf.layers.conv2d(conv3, filters=256, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='enc_conv4')
        conv4 = tf.layers.batch_normalization(conv4, training=self.is_train)
        conv4 = tf.nn.relu(conv4)
 
        flat = tf.layers.flatten(conv4, name='enc_flatten')
        self.z_mu = tf.layers.dense(flat, self.n_z, activation=None, name='enc_mu')
        self.z_log_sigma_sq = tf.layers.dense(flat, self.n_z, activation=None, name='enc_sigma')
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # decoder 
        dec1 = tf.layers.dense(self.z, 1024, activation=tf.nn.relu, name='dec_fc1')
        dec1 = tf.reshape(dec1, shape=tf.shape(conv4), name='dec_reshape')

        dconv1 = tf.layers.conv2d_transpose(dec1, filters=128, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='dec_dconv1') 
        dconv1 = tf.layers.batch_normalization(dconv1, training=self.is_train)
        dconv1 = tf.nn.relu(dconv1)

        dconv2 = tf.layers.conv2d_transpose(dconv1, filters=64, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='dec_dconv2') 
        dconv2 = tf.layers.batch_normalization(dconv2, training=self.is_train)
        dconv2 = tf.nn.relu(dconv2)

        dconv3 = tf.layers.conv2d_transpose(dconv2, filters=32, kernel_size=(4,4), strides=(2,2), padding='valid', activation=None, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='dec_dconv3') 
        dconv3 = tf.layers.batch_normalization(dconv3, training=self.is_train)
        dconv3 = tf.nn.relu(dconv3)

        self.x_hat = tf.layers.conv2d_transpose(dconv3, filters=3, kernel_size=(6,6), strides=(2,2), padding='valid', activation=tf.nn.sigmoid, kernel_initializer=winit, bias_initializer=tf.zeros_initializer(), name='dec_xhat')          


        # reconstruction loss: L2 loss / mse
        self.recon_loss = tf.reduce_mean(tf.nn.l2_loss(tf.abs(self.x_hat-self.input_x))) 

        # latent loss: KL divergence between p(z) and N(0,1)
        self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)) 

        self.total_loss = self.recon_loss + self.beta * self.latent_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

        return

    # run single step
    def run_single_step(self, x, beta, is_train):
        _, loss, recon_loss, latent_loss = self.sess.run([self.train_op, self.total_loss, self.recon_loss, self.latent_loss], feed_dict={self.x: x, self.beta: beta, self.is_train: is_train})
        return loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstruction(self, x, beta, is_train):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x, self.beta: beta, self.is_train: is_train})
        x_hat *= 255.0 
        return x_hat

    # z -> x
    def generator(self, z, beta, is_train):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z, self.beta: beta, self.is_train: is_train})
        return x_hat

    # x -> z
    def encode(self, x, is_train):
        z = self.sess.run(self.z, feed_dict={self.x: x, self.is_train: is_train})
        return z

    # vae trainable vars
    def get_vae_vars(self):
        vae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return vae_vars

#---------------------------------------------------------------------------------------
