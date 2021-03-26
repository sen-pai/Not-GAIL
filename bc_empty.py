import gym
import os
import torch
import numpy as np
import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc
from stable_baselines3.common import policies

import gym_minigrid
from gym_minigrid import wrappers

with open("empty_6.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env(
    "MiniGrid-Empty-Random-6x6-v0",
    n_envs=1,
    post_wrappers=[wrappers.FlatObsWrapper],
    post_wrappers_kwargs=[{}],
)

logger.configure("bc_empty_warmstart_logs")
bc_trainer = bc.BC(
    venv.observation_space,
    venv.action_space,
    is_image=False,
    expert_data=transitions,
    loss_type="original",
    policy_class=policies.ActorCriticPolicy,
)
bc_trainer.train(n_epochs=50)
bc_trainer.save_policy("bc_empty_warmstart_policy.pt")
