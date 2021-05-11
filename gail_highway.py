import copy
import os

import gym
import highway_env
from highway_configs import kinematic_highway_config

import numpy as np
import pickle5 as pickle
import torch
from gym_minigrid import wrappers
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger, util
from stable_baselines3 import PPO
from stable_baselines3.common import policies
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed



def make_env(rank, seed=0):
    def _init():
        env = gym.make("highway-v0")
        env.configure(kinematic_highway_config)
        env.reset()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

with open("highway_traj.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


venv = make_env(1)

base_ppo = PPO("MlpPolicy", venv, verbose=1, batch_size=64, n_steps=256)

logger.configure("gail_highway")

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=64,
    gen_algo=base_ppo,
    n_disc_updates_per_round=1,
)

gail_trainer.train(total_timesteps=int(10000))


gail_trainer.gen_algo.save("gail_highway")
