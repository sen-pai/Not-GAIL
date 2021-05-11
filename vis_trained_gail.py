import gym
import os
import numpy as np

import gym_minigrid
from gym_minigrid import wrappers
from imitation.util import util
from imitation.algorithms import adversarial, bc
from stable_baselines3 import PPO


venv = util.make_vec_env("MiniGrid-Empty-Random-6x6-v0", n_envs=1, post_wrappers= [wrappers.FlatObsWrapper],  post_wrappers_kwargs=[{}],)

# trained_policy = bc.reconstruct_policy("bc_empty_warmstart_policy.pt")
# trained_policy = PPO.load("ppo_minigrid_empty")
trained_policy = PPO.load("empty_gail_gen_5")

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
