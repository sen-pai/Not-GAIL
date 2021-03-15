import gym
import os
import numpy as np

from imitation.util import util
from imitation.algorithms import bc


env = gym.make('CartPole-v1')
venv = util.make_vec_env("CartPole-v1", n_envs=1)

bc_trained_policy = bc.reconstruct_policy("cartpole_bc.pth")


obs = venv.reset()
for i in range(1000):
    action, _state = bc_trained_policy.predict(obs, deterministic=True)
    obs, reward, done, info = venv.step(action)
    venv.render()
    if done:
      obs = venv.reset()
