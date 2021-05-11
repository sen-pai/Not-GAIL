import gym
import highway_env
import os

import numpy as np
from imitation.data.types import Trajectory
from stable_baselines3 import PPO
import pickle

from imitation.util import logger, util

from highway_configs import  kinematic_highway_config

env = gym.make("highway-v0")
# env = gym.make("roundabout-v0")

env.configure(kinematic_highway_config)
env.reset()

model = PPO.load("mlp_ppo_highway_multi_w_v")

traj_dataset = []
for traj in range(30):
    obs_list = []
    action_list = []
    obs = env.reset()
    print(obs, traj)
    obs_list.append(obs)
    for i in range(80):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        action_list.append(action)
        obs_list.append(obs)
        env.render()
        if done:
            break
    traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array([{} for i in action_list])))


with open('highway_traj.pkl', 'wb') as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
