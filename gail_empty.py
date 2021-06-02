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

with open("trajectories/lava_crossing9_closestlava.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


venv = util.make_vec_env(
    "MiniGrid-LavaCrossingS9N1-v0",
    n_envs=1,
    post_wrappers=[wrappers.FlatObsWrapper],
    post_wrappers_kwargs=[{}],
)
# venv = util.make_vec_env(
#     'MiniGrid-Empty-Random-6x6-v0', 
#     n_envs=1, 
#     post_wrappers= [wrappers.FullyObsWrapper, wrappers.FlatObsWrapper], 
#     post_wrappers_kwargs=[{},{}]
# )


base_ppo = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size=50, n_steps=50)

logger.configure("MiniGrid-KeyEmpty-v0")

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=60,
    gen_algo=base_ppo,
    n_disc_updates_per_round=2,
    normalize_reward=False,
    normalize_obs=False
)

total_timesteps = 60000
for i in range(10):
    gail_trainer.train(total_timesteps=total_timesteps//10)
    gail_trainer.gen_algo.save("gens/gail_gen_"+str(i))

    with open('discrims/gail_discrim'+str(i)+'.pkl', 'wb') as handle:
        pickle.dump(gail_trainer.discrim, handle, protocol=pickle.HIGHEST_PROTOCOL)
