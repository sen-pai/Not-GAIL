import gym
import os
import numpy as np
import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import anything_module, something_module
from stable_baselines3 import PPO

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

import gym_minigrid
from gym_minigrid import wrappers

with open("lava_ideal.pkl", "rb") as f:
    trajectories = pickle.load(f)

good_transitions = rollout.flatten_trajectories(trajectories)
good_transitions = rollout.flatten_trajectories(trajectories)

with open("running_into_lava.pkl", "rb") as f:
    bad_trajectories = pickle.load(f)

bad_transitions = rollout.flatten_trajectories(bad_trajectories)
bad_transitions = rollout.flatten_trajectories(bad_trajectories)


venv = util.make_vec_env('MiniGrid-LavaCrossingS9N1-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper])

logger.configure("lava_something_logs")

neg_gail_trainer = anything_module.AnythingGAIL(
    venv,
    expert_data=bad_transitions,
    expert_batch_size=32,
    gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(1e3)),
    neg_gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(1e3)),

)

something_gail_trainer = something_module.SomethingGAIL(
    venv,
    expert_data=good_transitions,
    expert_batch_size=32,
    gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(1e3)),
    anything_trainers= [neg_gail_trainer]
)


something_gail_trainer.train(int(2e4))

something_gail_trainer.anything_trainers[0].gen_algo.save("lava_some_any_gen")
something_gail_trainer.anything_trainers[0].neg_gen_algo.save("lava_some_any_neg_gen")

something_gail_trainer.gen_algo.save("lava_something_gen")
