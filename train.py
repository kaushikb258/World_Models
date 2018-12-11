import numpy as np
import sys
import tensorflow as tf

from utils import *
from vae import *
from rnn import *


IMG_PATH = "/media/kaushik/frames/"
learning_rate = 1e-4 # learning rate
nr = 82 # number of rows
nc = 96 # number of columns



train_from_scratch = True


# VAE params
nsteps_vae = 50000  # number of steps to train VAE
batch_size = 256 # batch size
n_z = 32 # latent vector
img_size = 64 



# MDN-RNN params
nsteps_rnn = 20 # number of epochs to train RNN
num_units = 256 # number of units in LSTM
num_gauss_mix = 5 # number of Gaussians in GMM
tau_sample = 1.0 #1.15 # tau parameter in the paper
prob_start = 0.5 # scheduled sampling start prob
prob_end = 0.0 # scheduled sampling end prob



#----------------------------------------------------------------------------------------


create_dirs()


with tf.Session() as sess:


  vae = VAE(sess, lr=learning_rate, batch_size=batch_size, n_z=n_z, nr=nr, nc=nc, img_size=img_size)

  rnn = MDN_RNN(sess, n_z, num_units, num_gauss_mix, learning_rate)

 
  vae_vars = vae.get_vae_vars()
  rnn_vars = rnn.get_rnn_vars()
 

  saver_vae = tf.train.Saver(vae_vars)
  saver_rnn = tf.train.Saver(rnn_vars)
  

  if (train_from_scratch == True):
      sess.run(tf.global_variables_initializer())
  else:
      saver_vae.restore(sess, "ckpt/vae/model")  
      print("restored vae_vars")
    
      saver_rnn.restore(sess, "ckpt/rnn/model")   
      print("restored rnn_vars")
   
#----------------------------------------------------------------------------------------
# train VAE

  print('-'*25 + '\n')
  print("train VAE ")



  for ns in range(nsteps_vae):
   
     # set warm-up parameter beta (ladder VAE)
     beta = float(ns)/float(nsteps_vae/2) * 1.0
     beta = np.clip(beta, a_min=0.0, a_max=1.0) 


     # sample mini-batch
     batch_frames = get_mini_batch_frames(IMG_PATH, batch_size)

     # train one step
     loss, recon_loss, latent_loss = vae.run_single_step(batch_frames, beta, is_train=True)

     print("ns: ", ns, "| loss: ", loss, "| recon_loss: ", recon_loss, "| latent_loss: ", latent_loss)
     f = open("performance_vae.txt", "a+")
     f.write(str(ns) + " " + str(loss) + " " + str(recon_loss) + " " + str(latent_loss) + '\n')  
     f.close()     



     # reconstruct image once every N steps
     if (ns % 200 == 0):
         ii = np.random.randint(low=0, high=batch_size, size=1)
         x = np.squeeze(batch_frames[ii],axis=0)
         x_hat = vae.reconstruction(np.expand_dims(x,axis=0), beta, is_train=True)
         x_hat = np.squeeze(x_hat, axis=0)    
         x_hat = scipy.misc.imresize(x_hat, (nr,nc,3), interp='bilinear')
         assert x.shape == x_hat.shape

         combined_image = np.zeros((nr,2*nc+10,3),dtype=np.float32) 
         combined_image[:,:nc,:] = x[:,:,:]
         combined_image[:,-nc:,:] = x_hat[:,:,:]

         file_name = "vae_samples/sample_" + str(ns) + ".png"
         scipy.misc.imsave(file_name, combined_image)

         # save model
         saver_vae.save(sess, 'ckpt/vae/model') 
         print("saved vae model ")


#----------------------------------------------------------------------------------------
# train MDN-RNN

  print('-'*25 + '\n')
  print("train MDN-RNN ")

  actions = np.load('actions.npy')    

  na = actions.shape[0]
  print("na: ", na)

  prob_ss = prob_start 


  for ep in range(nsteps_rnn):

    state_in = rnn.state_init
    avg_ep_loss = 0.0

    n_teacher_forcing = 0
    n_inference_mode = 0

    for i in range(na-1):

       # scheduled sampling probability
       prob_ss = prob_start - (prob_start - prob_end)* float(i + ep*(na-1))/float(nsteps_rnn*(na-1)) 
       prob_ss = max(prob_ss, prob_end)
       prob_ss = min(prob_ss, prob_start)
       
       # st and stp1 
       st, stp1 = get_st_and_stp1(IMG_PATH, i)     
       st = np.expand_dims(st, axis=0)
       stp1 = np.expand_dims(stp1, axis=0)

       # encode st -> zt; stp1 -> ztp1
       zt = vae.encode(st, is_train=True) #False) 
       ztp1 = vae.encode(stp1, is_train=True) #False) 

       # at 
       at = actions[i]
       at = np.expand_dims(at, axis=0)

       # get GMM params and state_out from MDN-RNN
       gauss_mu, gauss_sigma, gauss_pi, state_out = rnn.get_rnn_out(zt, at, state_in)

       # sample ztp1 from MDN-RNN
       ztp1_sample = sample_from_GMM(n_z, num_gauss_mix, gauss_mu, gauss_sigma, gauss_pi, tau_sample)
       ztp1_sample = np.expand_dims(ztp1_sample,axis=0)   

       assert ztp1.shape == ztp1_sample.shape

       # use scheduled sampling for training MDN-RNN
       if (np.random.rand() < prob_ss):
           # teacher forcing       
           ztp1_for_training = ztp1
           n_teacher_forcing += 1
       else:
           # inference mode
           ztp1_for_training = ztp1_sample 
           n_inference_mode += 1 

       loss = rnn.train(zt, at, ztp1_for_training, state_in)    

       

       state_in = state_out
       avg_ep_loss += loss

       if (i % 1000 == 0):
          print("i: ", i, "| loss: ", loss, "| n_teacher_forcing: ", n_teacher_forcing, "| n_inference_mode: ", n_inference_mode)
          print("prob_ss: ", prob_ss)


    print("% of training in teacher forcing mode: ", round(100.0*float(n_teacher_forcing)/float(n_teacher_forcing+n_inference_mode),2))
    print("% of training in inference mode: ", round(100.0*float(n_inference_mode)/float(n_teacher_forcing+n_inference_mode),2))
    

    avg_ep_loss /= np.float(na-1)   
    print("ep: ", ep, "| avg loss: ", avg_ep_loss)
    print('-'*10 + '\n')
    f = open("performance_rnn.txt", "a+")
    f.write(str(ep) + " " + str(avg_ep_loss) + '\n')  
    f.close()     

 
    # save model
    saver_rnn.save(sess, 'ckpt/rnn/model') 
    print("saved rnn model ")

#----------------------------------------------------------------------------------------


