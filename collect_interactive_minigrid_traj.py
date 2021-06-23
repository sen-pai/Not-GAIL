import os
import matplotlib.pyplot as plt
from imitation.data.types import Trajectory
import pickle5 as pickle

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

import torch
import msvcrt
from utils import env_utils

from imitation.util import util
import gym_custom

traj_dataset = []
obs_list = []
action_list = []
info_list = []


# encoder_model = torch.load("models/ae_minigrid_empty.pt")
# venv = env_utils.minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", flat = True, n_envs=1, partial=False, encoder=None)

venv = util.make_vec_env(
    'CoverAllTargetsDiscrete-v0',
    n_envs=1
    # post_wrappers=[wrap.FlatObsWrapper],
    # post_wrappers_kwargs=[{}],
)


cnt = 0
obs = venv.reset()
obs_list.append(obs.reshape(-1))
while cnt<10:

    venv.render()
    x = msvcrt.getwch()

    if x=='k':
        break
    elif x=='n':
        traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array(info_list)))
        obs_list = []
        action_list = []
        info_list = []

        obs = venv.reset()
        obs_list.append(obs.reshape(-1))
        cnt+=1
        print(cnt)
        continue
    elif x == 'a':
        action = 0
    elif x == 'd':
        action = 1
    elif x== 'n':
        break
    elif x=='w':
        action = 2
    else:
        action = int(x)


    obs, reward, done, info = venv.step([action])
    action_list.append(action)
    info_list.append(info)
    obs_list.append(obs.reshape(-1))
  
    print(action, obs.shape)

    if done:   
        print("done")
        # print(obs_list[0].shape)
        traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array(info_list)))

        obs_list = []
        action_list = []
        info_list = []

        obs = venv.reset()
        obs_list.append(obs.reshape(-1))
        cnt+=1
        print(cnt)


with open('traj_datasets/free_moving_discrete_circle.pkl', 'wb') as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('data/empty6x6_traj_collection_action_list.pkl', 'wb') as handle:
#     pickle.dump(act_l, handle, protocol=pickle.HIGHEST_PROTOCOL)