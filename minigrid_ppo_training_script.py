# import os
import numpy as np

import argparse

import matplotlib.pyplot as plt

import gym
import gym_minigrid
from gym_minigrid import wrappers
from numpy.core.fromnumeric import transpose

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    "-e",
    help="minigrid gym environment to train on",
    default="MiniGrid-LavaCrossingS9N1-v0",
)
parser.add_argument("--run", "-r", help="Run name", default="testing")
parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=1
)
parser.add_argument(
    "--timesteps", "-t", type=int, help="total timesteps to learn", default=1e5
)

parser.add_argument(
    "--nenvs", type=int, help="number of parallel environments to train on", default=1
)


parser.add_argument(
    "--lr",
    "-lr",
    type=float,
    help="learning rate, no need to change, default same as ppo",
    default=3e-4,
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


args = parser.parse_args()

def get_env(n_envs=args.nenvs):

    img_wrappers = lambda env: wrappers.ImgObsWrapper(wrappers.RGBImgObsWrapper(env))
    flat_wrapper = lambda env: wrappers.FlatObsWrapper(env)

    vec_env = make_vec_env(
        env_id=args.env,
        n_envs=n_envs,
        wrapper_class=flat_wrapper if args.flat else img_wrappers,
    )

    if args.flat:
        return vec_env
    return VecTransposeImage(vec_env)


train_env = get_env()
eval_env = get_env(1)

if args.show and not args.flat:
    plt.imshow(np.moveaxis(train_env.reset()[0], 0, -1))
    plt.show()

save_path = "./logs/" + args.env + "/" + args.run

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=save_path)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.98, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=None,
    log_path=save_path + "/eval_results",
    eval_freq=1000,
    callback_on_new_best=callback_on_best,
)
callback = CallbackList([checkpoint_callback, eval_callback])


policy_type = "MlpPolicy" if args.flat else "CnnPolicy"

model = PPO(policy=policy_type, env=train_env, verbose=1, seed=args.seed)
model.learn(total_timesteps=int(args.timesteps), callback=callback)
