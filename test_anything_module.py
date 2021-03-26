import gym
import os
import numpy as np
import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import anything_module
from stable_baselines3 import PPO


env = gym.make('CartPole-v1')

with open("cartpole_proper.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=1)

logger.configure("cartpole_neg_logs")

neg_gail_trainer = anything_module.AnythingGAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(960)),
    neg_gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(960)),

)

neg_gail_trainer.train(total_timesteps= int(1e5))


neg_gail_trainer.gen_algo.save("normal_gail_cartpole")
neg_gail_trainer.neg_gen_algo.save("neg_gail_cartpole")
