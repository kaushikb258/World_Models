from mpi4py import MPI
import tensorflow as tf
import gym
import numpy as np
import sys
import matplotlib.pyplot as plt

from utils import *
from vae import *
from rnn import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




# params for vae and mdn-rnn
learning_rate = 1e-4
n_z = 32
n_h = 256
nr = 82
nc = 96
batch_size = 256
img_size = 64
num_units = 256
num_gauss_mix = 5


# params for es
nepisodes = 16
start_iter = 0
nsteps = 500 
lr_es = 5e-3
sigma_perturb_init = 0.1
sigma_perturb_final = 0.01
anneal_steps = 500
tau_sample = 1.0

train_from_scratch = True

#--------------------------------------------------------------------------------------

class ES():
   
    def __init__(self, sess, lr, n_z, n_h, scope='es'):
      self.sess = sess
      self.lr = lr
      self.n_z = n_z
      self.n_h = n_h
      self.scope = scope

      self.es_model()

    def es_model(self):
      with tf.variable_scope(self.scope): 
         
          # tf placeholders
          self.zt = tf.placeholder(name='zt', dtype=tf.float32, shape=[self.n_z])  
          self.ht = tf.placeholder(name='ht', dtype=tf.float32, shape=[self.n_h])
          self.w_es = tf.placeholder(name='w_es', dtype=tf.float32, shape=[self.n_z + self.n_h, 3])
          self.b_es = tf.placeholder(name='b_es', dtype=tf.float32, shape=[3])         

          z_and_h = tf.concat(values=[self.zt, self.ht], axis=0) 
          self.at = tf.nn.xw_plus_b(tf.expand_dims(z_and_h,axis=0), self.w_es, self.b_es)

    def compute_at(self, zt, state_in, w_es, b_es):
       if (len(state_in) == 2):
         ct, ht = state_in[0][0], state_in[1][0]
       else: 
         print("wrong entry for state_in ", len(state_in))
         sys.exit() 
       ht = np.array(ht)

       at = self.sess.run([self.at], feed_dict={self.zt: np.squeeze(zt,axis=0), self.ht: ht, self.w_es: w_es, self.b_es: b_es})
       at = at[0][0] # what comes out of the previous line is at = [[]]  
       at[0] = np.tanh(at[0]) # steering: [-1, 1]
       at[1] = (np.tanh(at[1]) + 1.0)/2.0 # acceleration: [0, 1]
       at[2] = (np.tanh(at[2]) + 1.0)/2.0 # brake: [0, 1]

       # fix acceleration and brake
       at[1] = 0.25 + 0.75*at[1]
       at[2] = 0.4*at[2]
       return at  


#--------------------------------------------------------------------------------------

if (rank == 0):
    if (train_from_scratch):
       mu_es = 0.0
       sigma_es = 0.2
       w_es = np.random.normal(mu_es, sigma_es, (n_z + n_h, 3))
       w_es = np.array(w_es,dtype=np.float32)
       b_es = np.random.normal(mu_es, sigma_es, (3))
       b_es = np.array(b_es,dtype=np.float32)
    else:
       w_es = np.load("ckpt/es/w_es_" + str(start_iter-1) + ".npy")
       b_es = np.load("ckpt/es/b_es_" + str(start_iter-1) + ".npy")
else:
    w_es = np.empty((n_z + n_h, 3), dtype=np.float32)
    b_es = np.empty((3), dtype=np.float32)
  

comm.Bcast(w_es, root=0)
comm.Bcast(b_es, root=0)

comm.Barrier()
#--------------------------------------------------------------------------------------

# train ES


if (rank != 0):
  
   sess = tf.Session()

   vae = VAE(sess, lr=learning_rate, batch_size=batch_size, n_z=n_z, nr=nr, nc=nc, img_size=img_size)
   rnn = MDN_RNN(sess, n_z, num_units, num_gauss_mix, learning_rate)
 
         
   env = gym.make("CarRacing-v0")
  
   es = ES(sess, lr_es, n_z, n_h)


   vae_vars = vae.get_vae_vars()
   rnn_vars = rnn.get_rnn_vars()
   print(len(vae_vars), len(rnn_vars))

   saver_vae = tf.train.Saver(vae_vars)
   saver_rnn = tf.train.Saver(rnn_vars)

   sess.run(tf.global_variables_initializer())

   saver_vae.restore(sess, "ckpt/vae/model")  
   saver_rnn.restore(sess, "ckpt/rnn/model")   


   for k in range(start_iter, start_iter+nsteps):

     w_es = np.empty((n_z + n_h, 3), dtype=np.float32)
     b_es = np.empty((3), dtype=np.float32)
     sigma_perturb = np.empty((1), dtype=np.float32) 
     comm.Recv(sigma_perturb, source=0, tag=77)
     comm.Recv(w_es, source=0, tag=77)
     comm.Recv(b_es, source=0, tag=77)


     # perturb weights
     eps_w = np.random.normal(0.0, sigma_perturb, (n_z + n_h, 3))
     eps_b = np.random.normal(0.0, sigma_perturb, (3))
     eps_w = np.array(eps_w,dtype=np.float32)
     eps_b = np.array(eps_b,dtype=np.float32)
     w_es += eps_w
     b_es += eps_b

     comm.Send(eps_w, dest=0, tag=77)
     comm.Send(eps_b, dest=0, tag=77)

     avg_ep_reward = 0.0

     for _ in range(nepisodes):

        st = env.reset()
        st = st[:nr,:,:]
        done = False
        ep_reward = 0.0
        tsteps = 0
   
        state_in = rnn.state_init

        n_step_on_gas = np.random.randint(20, 50)

        stm1 = st 


        while not done:

          env.render()
    
          zt = vae.encode(np.expand_dims(st,axis=0), is_train=True) #False)           
          zt_to_controller = zt # initial value (may get over-written below)


          if (tsteps < n_step_on_gas):
             at = [0.0, 1.0, 0.0]
          else:

#             if (np.random.rand() < 0.5):
#               # use vae's output
#               zt_to_controller = zt
#             else:
#               # use ztp1 predicted from mdn-rnn in the previous time step
#               zt_to_controller = ztp1_sample 

             # use vae's output
             zt_to_controller = zt 

             at = es.compute_at(zt_to_controller, state_in, w_es, b_es)          

          # no movement in frames, step on gas!
          if (tsteps > 0 and np.array_equal(stm1,st)):
              print("states equal, step on gas! ", rank)
              at = [0.0, 1.0, 0.0] 

          
          # sample ztp1 from MDN-RNN (to be used in the next time step as zt_to_controller)
          # one could use zt instead of zt_to_controller in the following line -- maybe worth experimenting
          gauss_mu, gauss_sigma, gauss_pi, state_out = rnn.get_rnn_out(zt_to_controller, np.expand_dims(at,axis=0), state_in)
          ztp1_sample = sample_from_GMM(n_z, num_gauss_mix, gauss_mu, gauss_sigma, gauss_pi, tau_sample)
          ztp1_sample = np.expand_dims(ztp1_sample,axis=0)  


          stp1, r, done, info = env.step(at)

          stp1 = stp1[:nr,:,:]
          stm1 = st
          st = stp1
          ep_reward += r
          tsteps += 1
          ctp1, htp1 = state_out #[0]
          state_in =  [ctp1, htp1]    

          if (tsteps > 100):
            if (not is_car_on_road(stp1)):
              ep_reward -= 25.0 
              done = True 

          # congrats on coming this far!
          #if (tsteps == min(250+3*k,1000)):
          #    ep_reward += 25
          #    done = True
           

        print("rank: ", rank, "| ep_reward: ", ep_reward, "| k: ", k)
        avg_ep_reward += ep_reward
        

     avg_ep_reward /= np.float(nepisodes)
     avg_ep_reward = np.array(avg_ep_reward,dtype=np.float32)
  
     comm.Send(avg_ep_reward, dest=0, tag=77)
   

else:

   for k in range(start_iter, start_iter+nsteps): 
 
      sigma_perturb = sigma_perturb_init - float(k)/float(anneal_steps) * sigma_perturb_init 
      sigma_perturb = max(sigma_perturb, sigma_perturb_final) 
      sigma_perturb = np.array(sigma_perturb,dtype=np.float32)
 
      for i in range(1, size): 
         comm.Send(sigma_perturb, dest=i, tag=77)
         comm.Send(w_es, dest=i, tag=77)
         comm.Send(b_es, dest=i, tag=77)


      # receive eps_w and eps_b
      eps_w = np.empty((size-1,n_z+n_h,3), dtype=np.float32)
      eps_b = np.empty((size-1,3), dtype=np.float32)
      for i in range(1, size):
          data1 = np.empty((n_z+n_h,3), dtype=np.float32)
          data2 = np.empty((3), dtype=np.float32)
          comm.Recv(data1, source=i, tag=77)
          comm.Recv(data2, source=i, tag=77) 
          eps_w[i-1,:,:] = data1
          eps_b[i-1,:] = data2  


      avg_ep_reward_ = np.empty((size-1), dtype=np.float32)

      for i in range(1, size):
        data1 = np.empty((1), dtype=np.float32)
        comm.Recv(data1, source=i, tag=77)
        avg_ep_reward_[i-1] = data1    

      Fmin = np.amin(avg_ep_reward_)
      Fmax = np.amax(avg_ep_reward_) 
      Fmean = np.mean(avg_ep_reward_)
    
      F = (avg_ep_reward_ - np.mean(avg_ep_reward_))/np.std(avg_ep_reward_)

      F_eps_w = np.zeros((n_z+n_h,3),dtype=np.float32)
      F_eps_b = np.zeros((3),dtype=np.float32)
      for i in range(size-1):
        F_eps_w[:,:] += F[i] * eps_w[i,:,:]
        F_eps_b[:] += F[i] * eps_b[i,:] 

      # perform es update
      # we have sigma^2 in the denominator as the epsilon we have is actually N(0,1)*sigma
      w_es = w_es + lr_es * F_eps_w / np.float(size-1) / (sigma_perturb**2) 
      b_es = b_es + lr_es * F_eps_b / np.float(size-1) / (sigma_perturb**2) 

      print('-'*10 + '\n') 
      print("step: ", k, "| Fmin: ", Fmin, "| Fmax: ", Fmax, "| Fmean: ", Fmean)
      print(avg_ep_reward_)
      f = open("performance_es.txt", "a+")
      f.write(str(k) + " " + str(Fmin) + " " + str(Fmax) + " " + str(Fmean) + ' \n')  
      f.close()     

      print("saving es weights ")
      np.save('ckpt/es/w_es_' + str(k),w_es)
      np.save('ckpt/es/b_es_' + str(k),b_es)

      print("w_es: ", w_es)
      print("b_es: ", b_es)
  

#----------------------------------------------------------------------------

comm.Barrier()


if (rank == 0):
   print("trained controller using es")
   np.save('ckpt/es/w_es_' + str(k),w_es)
   np.save('ckpt/es/b_es_' + str(k),b_es)
  


