import pickle5 as pickle

import gym
import gym_minigrid
from stable_baselines3 import PPO
from gym.wrappers.frame_stack import FrameStack
from stable_baselines3.common import policies, vec_env

from imitation.data import buffer, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, util
from imitation.data import rollout

import msvcrt
import numpy as np

from bac_utils.env_utils import minigrid_render, minigrid_get_env
from BaC import bac_wrappers
from modules.rnn_discriminator import ActObsCRNN
from BaC.bac_rnn import BaCRNN

import torch

def cust_rew(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:

    return np.array([-1]*len(state))


venv = minigrid_get_env(
    'MiniGrid-KeyEmpty-6x6-v0',
    n_envs=1,
)


bac_class = ActObsCRNN(
    action_space=venv.action_space, observation_space=venv.observation_space
)

traj_dataset_path = "./traj_datasets/saved_testing.pkl"
with open(traj_dataset_path, "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

bac_trainer = BaCRNN(
    venv,
    eval_env=None,
    bc_trainer=None,
    bac_classifier=bac_class,
    expert_data=transitions
)

bac_trainer.bac_classifier.load_state_dict(torch.load("bac_weights/key_test.pt", map_location=torch.device('cpu')))


venv = bac_wrappers.RewardVecEnvWrapperRNN(
    venv, 
    reward_fn=bac_trainer.predict, 
    bac_reward_flag=True   # Whether to use new_rews(False) or old_rews-new_rews(True)
)

x = ""
while x != "n":
    venv.render()
    x = msvcrt.getwch()

    if x == "a":
        action = [0]
    elif x == "d":
        action = [1]
    elif x == "n":
        break
    elif x == "w":
        action = [2]
    elif x == "e":
        action = [3]
    else:
        action = [int(x)]

    n_obs, reward, done, info = venv.step(action)
    
    print(action, reward)
    obs = n_obs
    if done[0]:
        print("done")
        obs = venv.reset()
    print()