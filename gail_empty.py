import copy
import os

import gym
import gym_minigrid
import numpy as np
import pickle5 as pickle
import torch
from gym_minigrid import wrappers
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger, util
from stable_baselines3 import PPO
from stable_baselines3.common import policies

with open("empty_6.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


venv = util.make_vec_env(
    "MiniGrid-Empty-Random-6x6-v0",
    n_envs=1,
    post_wrappers=[wrappers.FlatObsWrapper],
    post_wrappers_kwargs=[{}],
)

base_ppo = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size=10, n_steps=100)

logger.configure("empty_stack4_normal_gail")

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=10,
    gen_algo=base_ppo,
    n_disc_updates_per_round=1,
)

gail_trainer.train(total_timesteps=int(10000))


gail_trainer.gen_algo.save("empty_stack4_gail_gen")
