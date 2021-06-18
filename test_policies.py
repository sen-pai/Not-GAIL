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
import gym_custom

import torch
from utils import env_utils


import msvcrt
import sys

# encoder_model = torch.load("models/ae_minigrid_empty.pt")
# venv = env_utils.minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", flat=True, n_envs=1, partial=False, encoder=None)

venv = util.make_vec_env('FreeMovingContinuous-v0', n_envs=1)

for i in range(10):
    #i = int(sys.argv[1])
    discrim_file = "gail_training_data/discrims/gail_discrim"+str(i)+".pkl"
    gen_policy = "gail_training_data/gens/gail_gen_"+str(i)

    trained_policy = PPO.load(gen_policy)
    with open(discrim_file, "rb") as f:
        discrim = pickle.load(f)
    
    print(discrim_file)
    
    obs = venv.reset()
    tot_rew = 0
    
    x = ''
    while x!='n':
        # action, _state = trained_policy.predict(obs, deterministic=True)
        venv.reset()
        venv.render()
        x = msvcrt.getwch()

        if x=='r':
            for _ in range(500):
                venv.render()
                action, _state = trained_policy.predict(obs, deterministic=True)
                n_obs, reward, done, info = venv.step(action)

                rew = th.sigmoid(-th.tensor(discrim.predict_reward_test(state = obs, action = action, next_state = n_obs, done = done)))
                # rew2 = discrim.mod_rew(state = obs, action = action, next_state = n_obs, done = done)
                tot_rew += rew
                print(action, rew)
                
                obs = copy.deepcopy(n_obs)
                if done:
                    print("done")
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

        rew = th.sigmoid(-th.tensor(discrim.predict_reward_train(state = obs, action = action, next_state = n_obs, done = done)))
        # rew2 = discrim.mod_rew(state = obs, action = action, next_state = n_obs, done = done)
        tot_rew += rew
        print(action, rew)

        obs = n_obs
        if done:   
            print("Total Reward:", tot_rew)
            print("done")
            obs = venv.reset()
            tot_rew = 0