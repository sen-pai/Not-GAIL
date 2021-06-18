import copy
import os

import gym
import gym_minigrid
import numpy as np
import pickle5 as pickle
import torch
from gym_minigrid import wrappers
from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util
from stable_baselines3 import PPO
from stable_baselines3.common import policies
import gym_custom

from utils import env_wrappers, env_utils

with open("traj_datasets/free_moving_circle.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


# encoder_model = torch.load("models/ae_minigrid_empty.pt")
# venv = env_utils.minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", flat=True, n_envs=1, partial=False, encoder=None)

# venv = env_utils.minigrid_get_env(
#     "MiniGrid-LavaCrossingS9N1-v0",
#     n_envs=1,
#     post_wrappers=[wrappers.FlatObsWrapper],
#     post_wrappers_kwargs=[{}],
# )
venv = util.make_vec_env(
    'FreeMovingContinuous-v0', 
    n_envs=1
    # post_wrappers= [wrappers.FullyObsWrapper, wrappers.FlatObsWrapper], 
    # post_wrappers_kwargs=[{},{}]
)

# bc_trainer = bc.BC(observation_space=venv.observation_space, action_space=venv.action_space, expert_data=transitions, policy_class = policies.ActorCriticPolicy)
# bc_trainer.train(n_epochs=200)
# bc_trainer.save_policy("delete")

base_ppo = PPO(policies.ActorCriticPolicy,venv, verbose=1, batch_size=100, n_steps=500)
# base_ppo.load(path = "delete", env=venv)

logger.configure("logs/MiniGrid-Empty-6x6-v0")

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
    gail_trainer.gen_algo.save("gail_training_data/gens/gail_gen_"+str(i))

    with open('gail_training_data/discrims/gail_discrim'+str(i)+'.pkl', 'wb') as handle:
        pickle.dump(gail_trainer.discrim, handle, protocol=pickle.HIGHEST_PROTOCOL)
