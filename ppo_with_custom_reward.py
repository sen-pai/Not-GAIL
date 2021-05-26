import pickle5 as pickle

import gym
import gym_minigrid
from gym_minigrid import wrappers as wrap
from stable_baselines3 import PPO
from gym.wrappers.frame_stack import FrameStack
from stable_baselines3.common import policies, vec_env

from imitation.data import buffer, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper, util




venv = util.make_vec_env(
    'MiniGrid-Empty-Random-6x6-v0',
    n_envs=1,
    post_wrappers=[wrap.FlatObsWrapper],
    post_wrappers_kwargs=[{}],
)


venv_buffering = wrappers.BufferingWrapper(venv)
venv_wrapped = vec_env.VecNormalize(
    venv_buffering,
    norm_reward=False,
    norm_obs=False,
)

with open("discrims/gail_discrim9.pkl", "rb") as f:
    discrim = pickle.load(f)
venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
    venv_wrapped, discrim.mod_rew
)

venv = vec_env.VecNormalize(
    venv_wrapped, norm_obs=False, norm_reward=False
)

model = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size=50, n_steps=50)#PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps= int(5e4))#, callback=eval_callback)

model.save("ppo_lava")