import gym
from gym import wrappers
import os
import numpy as np
import pickle

from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc, adversarial
from imitation.rewards import my_discrim_nets
from stable_baselines3 import PPO



env = gym.make("CarRacing-v0")
env = wrappers.resize_observation.ResizeObservation(env, shape= 64)
env = wrappers.gray_scale_observation.GrayScaleObservation(env, keep_dim = False)
env = wrappers.frame_stack.FrameStack(env, num_stack = 4)

with open("cartpole_proper.pkl", "rb") as f:
    trajectories = pickle.load(f)


transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env(env, n_envs=1)


logger.configure("carracing_gail_logs")

my_discrim_net = my_discrim_nets.ObsActCNN(venv.action_space, venv.observation_space)

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo= PPO("CNNPolicy", venv, verbose=1, n_steps= int(1e3)),
    discrim_kwargs=(discrim_net = my_discrim_net)
)
gail_trainer.train(total_timesteps= int(1e5))

gail_trainer.gen_algo.save("gail_carracing")
