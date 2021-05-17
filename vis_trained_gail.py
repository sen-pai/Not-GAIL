import gym
import os
import numpy as np

import gym_minigrid
from gym_minigrid import wrappers
from imitation.util import util
from imitation.algorithms import adversarial, bc
from imitation.data import wrappers as wr
from stable_baselines3 import PPO
from stable_baselines3.common import preprocessing, vec_env

import pickle5 as pickle
import copy
import torch as th

th.set_grad_enabled(False)

venv = util.make_vec_env('MiniGrid-LavaCrossing-Random-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper], post_wrappers_kwargs=[{}])


for i in range(10):
  discrim_file = "discrims/gail_discrim"+str(i)+".pkl"
  gen_policy = "gens/gail_gen_"+str(i)
  # gen_policy = "bc_empty_warmstart_policy.pt" #BC
  # gen_policy = "ppo_minigrid_empty" #PPO
  gen_policy = "ppo_for_testing"
  # gen_policy = "ppo_minigrid_empty_fully_obs"

  trained_policy = PPO.load(gen_policy)
  with open(discrim_file, "rb") as f:
      discrim = pickle.load(f)

  print(gen_policy, discrim_file)

  cont = ''
  while cont!='n':
    obs = venv.reset()
    tot_rew = 0
    for _ in range(20):
      venv.render()
      action, _state = trained_policy.predict(obs, deterministic=True)
      n_obs, reward, done, info = venv.step(action)


      rew = th.sigmoid(th.tensor(-discrim.predict_reward_test(state = obs, action = action, next_state = n_obs, done = done)))
      tot_rew += rew
      print(action, rew)
      obs = copy.deepcopy(n_obs)
      if done:
        print("done")
        break
    print("Total reward:", tot_rew)
    cont = input()