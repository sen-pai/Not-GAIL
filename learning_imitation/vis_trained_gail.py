import gym
import os
import numpy as np

from imitation.util import util
from imitation.algorithms import adversarial
from stable_baselines3 import PPO


env = gym.make('CartPole-v1')
venv = util.make_vec_env("CartPole-v1", n_envs=1)

gail_trained_policy = PPO.load("gail_cartpole")


obs = venv.reset()
for i in range(1000):
    action, _state = gail_trained_policy.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    venv.render()
    if done:
      obs = venv.reset()
