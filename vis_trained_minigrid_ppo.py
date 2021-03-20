import gym
import os
import numpy as np

from imitation.util import util
from imitation.algorithms import adversarial
from stable_baselines3 import PPO

import pickle5 as pickle

import gym_minigrid
from gym_minigrid import wrappers
from stable_baselines3 import PPO


env = gym.make('MiniGrid-Empty-Random-6x6-v0')
env = wrappers.FlatObsWrapper(env)


ppo_trained_policy = PPO.load("ppo_minigrid_empty")


obs = env.reset()
for i in range(1000):
    action, _state = ppo_trained_policy.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
