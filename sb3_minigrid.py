import gym
import os
import numpy as np
import pickle5 as pickle
import gym_minigrid
from gym_minigrid import wrappers
from stable_baselines3 import PPO


env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
env = wrappers.FlatObsWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps= int(1e5))

