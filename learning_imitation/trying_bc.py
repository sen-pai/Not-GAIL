import gym
import os
import numpy as np
import pickle

from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc


env = gym.make('CartPole-v1')

with open("cartpole_proper.pkl", "rb") as f:
    trajectories = pickle.load(f)


transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=1)


logger.configure("cartpole_bc_logs")
bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
bc_trainer.train(n_epochs=20)
bc_trainer.save_policy("cartpole_bc.pth")
