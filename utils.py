import glob
import PIL
from PIL import Image
import os
import scipy
import numpy as np






def get_mini_batch_frames(IMG_PATH, batch_size):

   img_path = glob.glob(IMG_PATH + '*.png')
   ii = np.random.randint(low=0, high=len(img_path), size=batch_size) 
   raw_frames = []   

   for i in ii:
     img = Image.open(img_path[i])
     img = np.array(img)
     raw_frames.append(img)

   raw_frames = np.array(raw_frames)
   return raw_frames




def get_st_and_stp1(IMG_PATH, i):

   img = Image.open(IMG_PATH + 'frame_' + str(i) + '.png')
   st = np.array(img)

   img = Image.open(IMG_PATH + 'frame_' + str(i+1) + '.png')
   stp1 = np.array(img)

   return st, stp1



def create_dirs():
   try:
     os.stat("ckpt")
   except:
     os.mkdir("ckpt")

   try:
     os.stat("ckpt/vae")
   except:
     os.mkdir("ckpt/vae")

   try:
     os.stat("ckpt/rnn")
   except:
     os.mkdir("ckpt/rnn")

   try:
     os.stat("ckpt/es")
   except:
     os.mkdir("ckpt/es")

   try:
     os.stat("vae_samples")
   except:
     os.mkdir("vae_samples")





# find if car on road
def is_car_on_road(im):

    assert im.shape == (82, 96, 3)   

    # car pixels
    car_pixels = [204, 0, 0]


    # find car pixels
    car = np.zeros((im.shape[0],im.shape[1]),dtype=np.int)
    car[np.where((im==car_pixels).all(axis=2))] = 1 


    # find bounding box
    B = np.argwhere(car==1)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) 
    
    xstart = max(xstart-5, 0)
    xstop = min(xstop+5, im.shape[1])
    ystart = max(ystart-5, 0)
    ystop = min(ystop+5, im.shape[0])

    # find number of road pixels in bounding box 
    nroad = 0
    for i in range(ystart, ystop):
     for j in range(xstart, xstop):
       if (im[i,j,0] == im[i,j,1] and im[i,j,0] == im[i,j,2]):
         if (im[i,j,0] > 90 and im[i,j,1] > 90 and im[i,j,2] > 90 and im[i,j,0] < 120 and im[i,j,1] < 120 and im[i,j,2] < 120): 
           nroad += 1

    if (nroad > 2):
       return True
    else:
       return False




def sample_from_GMM(nz, ngaussians, mu_, sigma_, pi_, tau):
    assert mu_.shape == (nz, ngaussians)
    assert sigma_.shape == (nz, ngaussians)
    assert pi_.shape == (nz, ngaussians)

    z = np.zeros((nz),dtype=np.float32) 

    for i in range(nz):
      assert np.abs(np.sum(pi_[i,:])-1.0) <= 1.0e-4, pi_[i,:] 
      zsum = 0.0
      for j in range(ngaussians):
         zsum += pi_[i,j] * np.random.normal(loc=mu_[i,j],scale=(sigma_[i,j]*np.sqrt(tau))) 
      z[i] = zsum      

    return z
