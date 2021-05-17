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

venv = util.make_vec_env('MiniGrid-LavaCrossing-Random-v0', n_envs=1, post_wrappers_kwargs=[{}], post_wrappers= [wrappers.FlatObsWrapper])


for i in range(20):
    discrim_file = "discrims/gail_discrim"+str(i)+".pkl"
    gen_policy = "gens/gail_gen_"+str(i)
    gen_policy = "ppo_for_testing"

    trained_policy = PPO.load(gen_policy)
    with open(discrim_file, "rb") as f:
        discrim = pickle.load(f)
    
    print(discrim_file)
    
    obs = venv.reset()
    tot_rew = 0
    
    x = ''
    while x!='n':
        # action, _state = trained_policy.predict(obs, deterministic=True)
        venv.render()
        x = msvcrt.getwch()

        if x=='r':
            for _ in range(20):
                venv.render()
                action, _state = trained_policy.predict(obs, deterministic=True)
                n_obs, reward, done, info = venv.step(action)

                rew = th.sigmoid(-th.tensor(discrim.predict_reward_test(state = obs, action = action, next_state = n_obs, done = done)))
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

        rew = th.sigmoid(-th.tensor(discrim.predict_reward_test(state = obs, action = action, next_state = n_obs, done = done)))-1
        tot_rew += rew
        print(action, rew)
        obs = n_obs
        if done:   
            print("Total Reward:", tot_rew)
            print("done")
            obs = venv.reset()
            tot_rew = 0
    
      
    