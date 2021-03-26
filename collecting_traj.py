import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from imitation.data.types import Trajectory
from stable_baselines3 import PPO
import pickle5 as pickle

import gym_minigrid
from gym_minigrid import wrappers
from imitation.util import logger, util
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from gym.wrappers.frame_stack import FrameStack

# env = wrappers.FlatObsWrapper(gym.make('MiniGrid-Empty-Random-6x6-v0'))
# env = gym.make('MiniGrid-Empty-Random-6x6-v0')
# env = wrappers.RGBImgObsWrapper(env)
# env = wrappers.ImgObsWrapper(env)


# venv = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.RGBImgObsWrapper, wrappers.ImgObsWrapper])
env = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper, FrameStack], post_wrappers_kwargs=[{}, {"num_stack":10}])

# env = VecTransposeImage(venv)
# env = VecTransposeImage(DummyVecEnv(env))

model = PPO.load("ppo_stack4_minigrid_empty")

print(env.reset().shape)
traj_dataset = []
for traj in range(30):
    obs_list = []
    action_list = []
    info_list = []
    obs = env.reset()
    print(obs.shape)
    obs_list.append(obs.reshape(-1))
    for i in range(500):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        action_list.append(action[0])
        info_list.append({})
        obs_list.append(obs.reshape(-1))
        # env.render()
        if done:
            break
    traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array(info_list)))


with open('empty_6_stack4.pkl', 'wb') as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
