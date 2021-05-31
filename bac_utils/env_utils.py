import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_minigrid
from gym_minigrid import wrappers

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage



def minigrid_get_env(env, n_envs, flat = False, env_kwargs={}):

    img_wrappers = lambda env: wrappers.ImgObsWrapper(wrappers.RGBImgObsWrapper(env))
    flat_wrapper = lambda env: wrappers.FlatObsWrapper(env)

    vec_env = make_vec_env(
        env_id=env,
        n_envs=n_envs,
        wrapper_class=flat_wrapper if flat else img_wrappers,
        env_kwargs=env_kwargs
    )

    if flat:
        return vec_env
    return VecTransposeImage(vec_env)



def minigrid_get_env_rew_times(env, n_envs, flat = False, env_kwargs={}):

    img_wrappers = lambda env: RewardTimes(wrappers.ImgObsWrapper(wrappers.RGBImgObsWrapper(env)))
    flat_wrapper = lambda env: wrappers.FlatObsWrapper(env)

    vec_env = make_vec_env(
        env_id=env,
        n_envs=n_envs,
        wrapper_class=flat_wrapper if flat else img_wrappers,
        env_kwargs=env_kwargs
    )

    if flat:
        return vec_env
    return VecTransposeImage(vec_env)


def minigrid_render(obs):
    plt.imshow(np.moveaxis(obs[0], 0, -1))
    plt.show()
    plt.close()



class RewardTimes(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env, factor = 100):
        super().__init__(env)
        self.factor = factor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward = reward * self.factor

        if done:
            print(reward)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
