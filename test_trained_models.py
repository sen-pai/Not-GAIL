import copy
import os

import gym
import gym_minigrid
import numpy as np
import pickle5 as pickle
import torch as th
from gym_minigrid import wrappers
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger, util
from imitation.rewards import common as rewards_common
from stable_baselines3 import PPO
from stable_baselines3.common import policies


import msvcrt
from bac_utils.env_utils import minigrid_render, minigrid_get_env
from BaC import bac_wrappers

import get_classifier

def cust_rew(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:

    return np.array([-1]*len(state))


venv = minigrid_get_env('MiniGrid-MidEmpty-Random-6x6-v0', n_envs=1)
#util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper], post_wrappers_kwargs=[{}])

bac_trainer = get_classifier.get_classifier(venv)

venv = bac_wrappers.RewardVecEnvWrapperRNN(
    venv, 
    reward_fn=bac_trainer.predict, 
    bac_reward_flag=True   # Whether to use new_rews(False) or old_rews-new_rews(True)
)

# venv = bac_wrappers.RewardVecEnvWrapper(
#     venv, cust_rew
# )

policy = "models/ppo_empty_normal"

trained_policy = PPO.load(policy)

obs = venv.reset()
tot_rew = 0

x = ''
while x!='n':
    venv.render()
    x = msvcrt.getwch()

    if x=='r':
        for _ in range(20):
            venv.render()
            action, _state = trained_policy.predict(obs, deterministic=True)
            n_obs, reward, done, info = venv.step(action)

            
            print(action, reward)
            obs = n_obs
            if done:
                print("Total Reward:", tot_rew)
                print("done")
                obs = venv.reset()
                tot_rew = 0
                break
        continue
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

    n_obs, reward, done, info = venv.step(action)

    print(action, reward, venv.get_attr("agent_pos"))

    obs = n_obs
    if done:   
        print("Total Reward:", tot_rew)
        print("done")
        obs = venv.reset()
        tot_rew = 0