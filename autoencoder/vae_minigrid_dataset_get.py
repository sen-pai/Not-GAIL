import gym
import os
import numpy as np

import gym_minigrid
from gym_minigrid import wrappers
from imitation.util import util
from imitation.algorithms import adversarial, bc
from stable_baselines3 import PPO
from torchvision import datasets
import msvcrt

import matplotlib.pyplot as plt


import sys
sys.path.append('C:/Users/Shivin/Desktop/RLPaper/nogail/Not-GAIL')

from utils.env_utils import minigrid_get_env
import pickle5 as pickle
import torch


model = torch.load("models/ae_minigrid_empty.pt")
decode = True

if decode:
    venv = minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", n_envs=1, partial=False, encoder=model)
else:
    venv = minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", n_envs=1, partial=False, encoder=False)


dataset = []
obs = venv.reset()
for i in range(1000):

    # action = [venv.action_space.sample()]
    # while action[0]>=3:
    #     action = [venv.action_space.sample()]
    # print(action)
    
    x = msvcrt.getwch()
    if x == 'n':
        break
    elif x == 'a':
        action = [0]
    elif x == 'd':
        action = [1]
    elif x== 'n':
        break
    elif x=='w':
        action = [2]
    else:
        action = [int(x)]

    print(action)
    obs, reward, done, info = venv.step(action)
    # print(obs, obs.shape)

    if decode:
        recon = model.decode(obs)
        print(recon.shape)
        plt.imshow(np.transpose(recon, (1, 2, 0)))
        plt.show()
    else: 
        venv.render()

    dataset.append(obs[0])
    print(len(dataset))

dataset = np.array(dataset)
print(dataset.shape)

with open('data/ae_minigrid_empty_obs'+'.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)