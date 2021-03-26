import gym
import os
import numpy as np

import gym_minigrid
from gym_minigrid import wrappers
from imitation.util import util
from imitation.algorithms import adversarial, bc
from stable_baselines3 import PPO


venv = util.make_vec_env('MiniGrid-LavaCrossingS9N1-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper])

trained_policy = bc.reconstruct_policy("lava_bc_warmstart.pt")
# trained_policy = PPO.load("ppo_minigrid_empty")
# gail_trained_policy = PPO.load("something_gen")



obs = venv.reset()
for i in range(1000):
    action, _state = trained_policy.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    print(action)
    venv.render()
    if done:
      obs = venv.reset()
      print("done")
