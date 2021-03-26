import gym
import os
import numpy as np
import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import anything_module, something_module
from stable_baselines3 import PPO


env = gym.make('CartPole-v1')

with open("cartpole_worst.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=1)

logger.configure("cartpole_something_logs")

neg_gail_trainer = anything_module.AnythingGAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(1e3)),
    neg_gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(1e3)),

)

something_gail_trainer = something_module.SomethingGAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo= PPO("MlpPolicy", venv, verbose=1, n_steps= int(1e3)),
    anything_trainers= [neg_gail_trainer]
)


something_gail_trainer.train(int(2e4))

something_gail_trainer.anything_trainers[0].gen_algo.save("some_any_gen")
something_gail_trainer.anything_trainers[0].neg_gen_algo.save("some_any_neg_gen")

something_gail_trainer.gen_algo.save("something_gen")
