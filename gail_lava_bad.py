import gym
import os
import torch
import numpy as np
import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import adversarial, bc
from imitation.policies.base import FeedForward32Policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import gym_minigrid
from gym_minigrid import wrappers

with open("lava_ideal.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


venv = util.make_vec_env('MiniGrid-LavaCrossingS9N1-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper])


base_ppo = PPO("MlpPolicy", venv, verbose=1, n_steps= int(960))


logger.configure("lava_warmstart_bc_logs")
bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
bc_trainer.policy = base_ppo.policy
bc_trainer.train(n_epochs=100)
bc_trainer.policy.save("ideal_bcwarm_start")



logger.configure("ideal_lava_normal_gail")

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo= base_ppo

)

gail_trainer.train(total_timesteps= int(1e5))


gail_trainer.gen_algo.save("normal_gail_ideal_lava")
