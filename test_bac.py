from utils.env_utils import minigrid_render, minigrid_get_env
import os, time
import numpy as np

import argparse

import matplotlib.pyplot as plt


import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc

import gym
import gym_minigrid


from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy


from modules.cnn_discriminator import ActObsCNN
from imitation.algorithms import bc
import msvcrt
from BaC.bac import BaC

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    "-e",
    help="minigrid gym environment to train on",
    default="MiniGrid-LavaCrossingS9N1-v0",
)
parser.add_argument("--run", "-r", help="Run name", default="testing")

parser.add_argument(
    "--save-name", "-s", help="BC weights save name", default="saved_testing"
)

parser.add_argument("--traj-name", "-t", help="Run name", default="saved_testing")


parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=1
)

parser.add_argument(
    "--nenvs", type=int, help="number of parallel environments to train on", default=1
)

parser.add_argument(
    "--nepochs", type=int, help="number of epochs to train till", default=50
)


parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--show", default=False, help="See a sample image obs", action="store_true",
)

parser.add_argument(
    "--vis-trained",
    default=False,
    help="Render 10 traj of trained BC",
    action="store_true",
)


args = parser.parse_args()


train_env = minigrid_get_env(args.env, args.nenvs, args.flat)

traj_dataset_path = "./traj_datasets/" + args.traj_name + ".pkl"

print(f"Expert Dataset: {args.traj_name}")

with open(traj_dataset_path, "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

bac_class = ActObsCNN(
    action_space=train_env.action_space, observation_space=train_env.observation_space
).to("cuda")

bac_trainer = BaC(
    train_env,
    eval_env=None,
    bc_trainer=None,
    bac_classifier=bac_class,
    expert_data=transitions,
)

bac_trainer.train_bac_classifier()


x = ""
while x != "n":
    train_env.render()
    x = msvcrt.getwch()

    if x == "a":
        action = [0]
    elif x == "d":
        action = [1]
    elif x == "n":
        break
    elif x == "w":
        action = [2]
    else:
        action = [int(x)]

    n_obs, reward, done, info = train_env.step(action)

    rew = bac_trainer.predict(
        np.expand_dims(n_obs[0], axis=0), np.expand_dims(np.array(action[0]), axis=0),
    ).data

    if rew>0.001:
        rew = - rew
    else:
        rew = 0
    print(action, rew)
    obs = n_obs
    if done:
        print("done")
        obs = train_env.reset()
        tot_rew = 0
