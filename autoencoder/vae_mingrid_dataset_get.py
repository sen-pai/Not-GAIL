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


from utils.env_utils import minigrid_get_env
import pickle5 as pickle


import torch


model = torch.load("models/ae_minigrid_empty.pt")
venv = minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", n_envs=1, partial=False, encoder=model)

decode = True


dataset = []
obs = venv.reset()
for i in range(500):

    # action = [venv.action_space.sample()]
    # while action[0]>=3:
    #     action = [venv.action_space.sample()]
    # print(action)
    
    x = msvcrt.getwch()
    if x == 'n':
        break
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

dataset = np.array(dataset)
print(dataset.shape)

with open('data/minigrid_empty_obs'+'.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)