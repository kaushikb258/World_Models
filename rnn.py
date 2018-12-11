import numpy as np
import tensorflow as tf
import sys


class MDN_RNN():
    
     def __init__(self, sess, n_z, num_units, num_gauss_mix, learning_rate, scope='RNN'):
        self.sess = sess
        self.n_z = n_z  # 32 
        self.num_units = num_units # 256
        self.num_gauss_mix = num_gauss_mix # 5 
        self.lr = learning_rate
        self.scope = scope
          
        self.build_net()
        self.optimize_fn()
 

     def build_net(self):

       with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

        # tf placeholders
        self.zt = tf.placeholder(name='zt', dtype=tf.float32, shape=[1, self.n_z])  
        self.at = tf.placeholder(name='at', dtype=tf.float32, shape=[1, 3])
        self.ztp1 = tf.placeholder(name='ztp1', dtype=tf.float32, shape=[1, self.n_z])

        # lstm
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units, state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c]) 
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)  

        zt_and_at = tf.concat(values=[self.zt, self.at], axis=1) 
        zt_and_at = tf.expand_dims(zt_and_at, [0])
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, zt_and_at, initial_state=state_in, time_major=False, dtype=tf.float32)

        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, self.num_units])

        # 32 outputs * 5 gaussians * 3 (3 because of pi, mu, sigma) 
        # = 480           
        nout = self.n_z * self.num_gauss_mix * 3
 
        output_w = tf.get_variable("output_w", [self.num_units, nout]) 
        output_b = tf.get_variable("output_b", [nout])  
 
        rnn_out = tf.reshape(rnn_out, [-1, self.num_units])   
        rnn_out = tf.nn.xw_plus_b(rnn_out, output_w, output_b)
        rnn_out = tf.reshape(rnn_out, [-1, self.num_gauss_mix*3])

        # rnn_out must be of shape [32, 5*3]       
        assert rnn_out.shape.as_list() == [32, 15], rnn_out.shape.as_list() 


        def mdn_coeff(out):
           logpi, mu, logstd = tf.split(out, 3, 1)
           return logpi, mu, logstd
       
        gauss_logpi, gauss_mu, gauss_logstd = mdn_coeff(rnn_out) 


        # pi in the GMM, normalize it to make it sum to 1
        # also subtract max value to avoid exp of large numbers which may blow up!
        max_logpi = tf.reduce_max(gauss_logpi, 1, keep_dims=True)
        gauss_logpi = tf.subtract(gauss_logpi, max_logpi)  
        gauss_pi = tf.exp(gauss_logpi)
        sum_pi = tf.reduce_sum(gauss_pi, 1, keep_dims=True)
        gauss_pi = tf.divide(gauss_pi, sum_pi)

        # sigma in the GMM
        gauss_sigma = tf.exp(gauss_logstd) 


        self.gauss_mu = gauss_mu
        self.gauss_sigma = gauss_sigma
        self.gauss_pi = gauss_pi

        # self.gauss_pi, self.gauss_mu, self.gauss_sigma must be of shape [32, 5]
        assert self.gauss_mu.shape.as_list() == [self.n_z, self.num_gauss_mix], self.gauss_mu.shape.as_list()
        assert self.gauss_sigma.shape.as_list() == [self.n_z, self.num_gauss_mix], self.gauss_sigma.shape.as_list()
        assert self.gauss_pi.shape.as_list() == [self.n_z, self.num_gauss_mix], self.gauss_pi.shape.as_list()



     def optimize_fn(self):

       with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

        # gaussian function
        def normal_dist(x, mu, sigma): 
          result = tf.divide(tf.subtract(x, mu), sigma)
          result = -tf.square(result)/2.0 
          result = tf.divide(tf.exp(result), sigma)
          result = tf.divide(result, np.sqrt(2.0*np.pi)) 
          return result
 
        # gaussian mixture model loss function (log likelihood)
        def loss_function(gauss_pi, gauss_sigma, gauss_mu, y):
          result = normal_dist(y, gauss_mu, gauss_sigma)
          result = tf.reduce_sum(tf.multiply(result, gauss_pi), 1, keep_dims=True)
          result = tf.reduce_mean(-tf.log(result + 1.0e-8)) # avoid log zero
          return result 
           

        # reshape target
        target_z = tf.reshape(self.ztp1, [-1, 1])
        assert target_z.shape.as_list() == [32, 1], target_z.shape.as_list()

        # loss op
        self.loss = loss_function(self.gauss_pi, self.gauss_sigma, self.gauss_mu, target_z)

        optim = tf.train.AdamOptimizer(self.lr) 
        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        gv = optim.compute_gradients(self.loss, var_list=rnn_vars)
        grad_clip = 10.0
        clipped_gv = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gv]
        self.train_op = optim.apply_gradients(clipped_gv, name='train_rnn')             
          

     # train RNN one step
     def train(self, zt, at, ztp1, state_in):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.zt: zt, self.at: at, self.ztp1: ztp1, self.state_in[0]: state_in[0], self.state_in[1]: state_in[1]})
        return loss


     # get GMM params and state_out for RNN
     def get_rnn_out(self, zt, at, state_in):
        gauss_mu, gauss_sigma, gauss_pi, state_out = self.sess.run([self.gauss_mu, self.gauss_sigma, self.gauss_pi, self.state_out], feed_dict={self.zt: zt, self.at: at, self.state_in[0]: state_in[0], self.state_in[1]: state_in[1]})
        return gauss_mu, gauss_sigma, gauss_pi, state_out


     # rnn trainable vars
     def get_rnn_vars(self):
        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return rnn_vars
 

