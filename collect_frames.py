import numpy as np
import sys
import os
import gym
from PIL import Image
import matplotlib.pyplot as plt

from utils import *




try:
    os.stat("frames")
except:
    os.mkdir("frames")


env = gym.make("CarRacing-v0")

episodes = 10000
    

#------------------------------------------------------------------------------------------



ii = 0
act = []


for ep in range(episodes):

  s = env.reset()

  tstep = 0

  ep_reward = 0
  
  while True:
 
    tstep += 1

    steer = np.random.uniform(low=-1.0, high=1.0)
    acc = np.random.uniform(low=0.0, high=1.0)
    br = np.random.uniform(low=0.0, high=0.2)
    actions = [steer, acc, br]  

    env.render()

    if (tstep > 50):
      act.append(actions)
     
      im = Image.fromarray(s[:82,:,:])
      im.save("frames/frame_" + str(ii) + ".png")
      ii += 1  

      
    next_s, reward, done, info = env.step(actions)

    ep_reward += reward

    if (tstep > 50):
       if (not is_car_on_road(next_s[:82,:,:])):
          done = True 
      

    if (done):
      print("episode: ", ep, "episode reward: ", ep_reward)
      break
    else:
      s = next_s



act = np.array(act)
np.save("actions", act)
print(act.shape)

