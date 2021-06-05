from libraries.imitation.src.imitation.data.types import Trajectory
from bac_utils.env_utils import seed_everything, minigrid_get_env
import os, time
import numpy as np

import argparse

import matplotlib.pyplot as plt
import torch

import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc

import gym
import gym_minigrid


from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy


from modules.cnn_discriminator import ActObsCNN, ObsCNN

from imitation.algorithms import bc
import msvcrt
from BaC.bac_cnn import BaC_CNN
from BaC.ae_trainer import VAE_trainer
from modules.cnn_autoencoder import CNN_VAE

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

parser.add_argument("--not-traj-name", "-nt", help="Run name", default="NA")


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
    "--triplet-epochs",
    "-te",
    type=int,
    help="number of triplet epochs to train",
    default=60,
)

parser.add_argument(
    "--classify-epochs",
    "-ce",
    type=int,
    help="number of epochs to train till",
    default=80,
)


parser.add_argument(
    "--warm", "-w", default=False, help="Expert warm start", action="store_true",
)

parser.add_argument(
    "--flat",
    "-f",
    default=False,
    help="Partially Observable FlatObs or Fully Observable Image ",
    action="store_true",
)

parser.add_argument(
    "--bc", default=False, help="Train BC", action="store_true",
)


parser.add_argument(
    "--vis-trained",
    default=False,
    help="Render 10 traj of trained BC",
    action="store_true",
)

parser.add_argument(
    "--load", "-l", default=False, help="Check loaded", action="store_true",
)

parser.add_argument(
    "--ae", "-ae", default=False, help="train VAE", action="store_true",
)

args = parser.parse_args()
print(args)

not_expert_provided = False
seed_everything(0)

env_kwargs = {}
if "FourRooms" in args.env:
    env_kwargs = {"agent_pos": (3, 3), "goal_pos": (15, 15)}

train_env = minigrid_get_env(args.env, args.nenvs, args.flat, env_kwargs)

traj_dataset_path = "./traj_datasets/" + args.traj_name + ".pkl"

print(f"Expert Dataset: {args.traj_name}")

with open(traj_dataset_path, "rb") as f:
    trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)

if args.not_traj_name != "NA":
    not_expert_provided = True
    with open("./traj_datasets/" + args.not_traj_name + ".pkl", "rb") as f:
        not_trajectories = pickle.load(f)
        not_transitions = rollout.flatten_trajectories(not_trajectories)


if args.ae:
    ae_class = CNN_VAE().to("cuda")
    vae_trainer = VAE_trainer(train_env, ae_class)
    vae_trainer.train()
else: 
    ae_class = None

bac_class = ActObsCNN(
    action_space=train_env.action_space, observation_space=train_env.observation_space, cnn_feature_extractor = ae_class
).to("cuda")

# if args.bc:
#     policy_type = ActorCriticPolicy if args.flat else ActorCriticCnnPolicy

#     transitions = rollout.flatten_trajectories(trajectories)

#     bc_trainer = bc.BC(
#         train_env.observation_space,
#         train_env.action_space,
#         is_image=False,
#         expert_data=transitions,
#         loss_type="original",
#         policy_class=policy_type,
#     )
#     bc_trainer.train(n_epochs=80)


bac_trainer = BaC_CNN(
    train_env,
    bc_trainer=None,
    bac_classifier=bac_class,
    expert_data=transitions,
    not_expert_data=not_transitions if not_expert_provided else None,
    nepochs=args.nepochs,
)


if args.load:
    bac_trainer.bac_classifier.load_state_dict(
        torch.load("bac_weights/" + args.save_name + ".pt")
    )
else:
    if args.warm:
        bac_trainer.expert_warmstart()

    bac_trainer.train()

    bac_trainer.save("bac_weights", args.save_name + ".pt")

    # for traj in trajectories:
    #     print("expert: ", bac_trainer.predict(traj).item())

    # if not_expert_provided:
    #     for traj in not_trajectories:
    #         print("not expert: ", bac_trainer.predict(traj).item())


x = ""

obs= train_env.reset()
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
    elif x == "e":
        action = [3]
    elif x == "p":
        obs = train_env.reset()
    else:
        action = [int(x)]

    n_obs, reward, done, info = train_env.step(action)

    rew = bac_trainer.predict(np.expand_dims(n_obs[0], axis=0), np.expand_dims(np.array(action[0]), axis=0)).item()

    if rew > 0.01:
        rew = -rew
    else:
        rew = 0
    print(action, rew)
    obs = n_obs
    if done:
        print("done")
        obs = train_env.reset()
        obs_list = [obs[0]]
        action_list = []
        tot_rew = 0
