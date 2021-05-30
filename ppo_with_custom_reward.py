import pickle5 as pickle

import gym
import gym_minigrid
from gym_minigrid import wrappers as wrap
from stable_baselines3 import PPO
from gym.wrappers.frame_stack import FrameStack
from stable_baselines3.common import policies, vec_env

from imitation.data import buffer, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper, util

from bac_utils.env_utils import minigrid_render, minigrid_get_env
import numpy as np

from bac_utils.env_utils import minigrid_render, minigrid_get_env
from BaC import bac_wrappers

import torch

def cust_rew(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:

    return np.array([-1]*len(state))


venv = minigrid_get_env('MiniGrid-Empty-Random-6x6-v0',n_envs = 1)
# util.make_vec_env(
#     'MiniGrid-Empty-Random-6x6-v0',
#     n_envs=1,
#     post_wrappers=[wrap.FlatObsWrapper],
#     post_wrappers_kwargs=[{}],
# )


# venv_buffering = wrappers.BufferingWrapper(venv)
# venv_wrapped = vec_env.VecNormalize(
#     venv_buffering,
#     norm_reward=False,
#     norm_obs=False,
# )

# with open("discrims/gail_discrim9.pkl", "rb") as f:
#     discrim = pickle.load(f)

venv = bac_wrappers.RewardVecEnvWrapper(
    venv, cust_rew
)

model = PPO(policies.ActorCriticCnnPolicy, venv, verbose=1, batch_size=50, n_steps=50)#PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps= int(3e4))#, callback=eval_callback)

model.save("models/ppo_empty_normal")