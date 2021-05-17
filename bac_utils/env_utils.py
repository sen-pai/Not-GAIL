import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_minigrid
from gym_minigrid import wrappers

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage



def minigrid_get_env(env, n_envs, flat = False):

    img_wrappers = lambda env: wrappers.ImgObsWrapper(wrappers.RGBImgObsWrapper(env))
    flat_wrapper = lambda env: wrappers.FlatObsWrapper(env)

    vec_env = make_vec_env(
        env_id=env,
        n_envs=n_envs,
        wrapper_class=flat_wrapper if flat else img_wrappers,
    )

    if flat:
        return vec_env
    return VecTransposeImage(vec_env)


def minigrid_render(obs):
    plt.imshow(np.moveaxis(obs[0], 0, -1))
    plt.show()
    plt.close()