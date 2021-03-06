from libraries.imitation.src.imitation.data.types import Trajectory
from utils.env_utils import minigrid_render, minigrid_get_env
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


from modules.rnn_discriminator import ActObsCRNN
from imitation.algorithms import bc
import msvcrt
from BaC.bac_rnn import BaCRNN

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
    "--show", default=False, help="See a sample image obs", action="store_true",
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
    "--load",
    "-l",
    default=False,
    help="Check loaded",
    action="store_true",
)


args = parser.parse_args()
print(args)


env_kwargs = {}
# if "FourRooms" in args.env:
#     env_kwargs = {"agent_pos": (3, 3), "goal_pos": (15, 15)}

train_env = minigrid_get_env(args.env, args.nenvs, args.flat,  env_kwargs)

traj_dataset_path = "./traj_datasets/" + args.traj_name + ".pkl"

print(f"Expert Dataset: {args.traj_name}")

with open(traj_dataset_path, "rb") as f:
    trajectories = pickle.load(f)


# with open("./traj_datasets/not_lava.pkl", "rb") as f:
#     bad_trajectories = pickle.load(f)

# transitions = rollout.flatten_trajectories(trajectories)

bac_class = ActObsCRNN(
    action_space=train_env.action_space, observation_space=train_env.observation_space
).to("cuda")


if args.bc:
    policy_type = ActorCriticPolicy if args.flat else ActorCriticCnnPolicy

    transitions = rollout.flatten_trajectories(trajectories)

    bc_trainer = bc.BC(
        train_env.observation_space,
        train_env.action_space,
        is_image=False,
        expert_data=transitions,
        loss_type="original",
        policy_class=policy_type,
    )
    bc_trainer.train(n_epochs=20)

    # for traj in range(10):
    #     obs = train_env.reset()
    #     train_env.render()
    #     for i in range(40):
    #         action, _ = bc_trainer.policy.predict(obs, deterministic=True)

    #         obs, reward, done, info = train_env.step(action)
    #         train_env.render()
    #         if done:
    #             break
    




bac_trainer = BaCRNN(
    train_env,
    eval_env=None,
    bc_trainer=bc_trainer if args.bc else None,
    bac_classifier=bac_class,
    expert_data=trajectories,
    # non_expert_data=bad_trajectories,
    nepochs=args.nepochs,
)

if args.load:
    bac_trainer.bac_classifier.load_state_dict(torch.load("bac_weights/"+ args.save_name+".pt"))
else:
    if args.warm:
        bac_trainer.expert_warmstart()

    bac_trainer.train_bac_classifier()

    bac_trainer.save("bac_weights", args.save_name+".pt")

    for traj in trajectories:
        print(bac_trainer.predict(traj).item())

obs_list = []
action_list = []
x = ""

obs_list.append(train_env.reset()[0])
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
    elif x =="p":
        obs_list = [train_env.reset()[0]]
        action_list = []
    elif x =="m":
        obs_list = [n_obs[0]]
        action_list = []
    
    else:
        action = [int(x)]
    

    action_list.append(action[0])
    n_obs, reward, done, info = train_env.step(action)

    obs_list.append(n_obs[0])
    rew = bac_trainer.predict(
        Trajectory(
            obs=np.array(obs_list),
            acts=np.array(action_list),
            infos=np.array([{} for i in action_list]),
        )
    ).item()

    if rew > 0.001:
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




# obs_list = []
# action_list = []
# x = ""

# obs_list.append(train_env.reset()[0])
# while x != "n":
#     train_env.render()
#     x = msvcrt.getwch()

#     if x == "a":
#         action = [0]
#     elif x == "d":
#         action = [1]
#     elif x == "n":
#         break
#     elif x == "w":
#         action = [2]
#     elif x =="p":
#         obs_list = [train_env.reset()[0]]
#         action_list = []
#     elif x =="m":
#         obs_list = [n_obs[0]]
#         action_list = []
    
#     else:
#         action = [int(x)]
    

#     action_list.append(action[0])
#     n_obs, reward, done, info = train_env.step(action)

#     obs_list.append(n_obs[0])
#     rew = bac_trainer.predict(
#         Trajectory(
#             obs=np.array([n_obs[0]]),
#             acts=np.array(action),
#             infos=np.array([{}]),
#         )
#     ).data

#     if rew > 0.001:
#         rew = -rew
#     # else:
#     #     rew = 0
#     print(action, rew)
#     obs = n_obs
#     if done:
#         print("done")
#         obs = train_env.reset()
#         obs_list = [obs[0]]
#         action_list = []
#         tot_rew = 0
