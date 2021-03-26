import gym
import os
import torch
import numpy as np
import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import adversarial, bc
from imitation.policies.base import FeedForward32Policy
from stable_baselines3.common import policies
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from gym.wrappers.frame_stack import FrameStack
import copy
import gym_minigrid
from gym_minigrid import wrappers

with open("empty_6_stack4.pkl", "rb") as f:
    trajectories = pickle.load(f)

#call after framestack or dont call at all :(
class FullFlatWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape = (10956,), dtype=self.observation_space.dtype)

    def observation(self, obs):
        # print(obs)
        # print(np.stack(obs).shape)
        return np.stack(obs).reshape(-1)


# print(trajectories)
transitions = rollout.flatten_trajectories(trajectories)


# venv = util.make_vec_env('MiniGrid-LavaCrossingS9N1-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper])
venv = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.FlatObsWrapper, FrameStack, FullFlatWrapper], post_wrappers_kwargs=[{}, {"num_stack":4}, {}])

# venv = util.make_vec_env('MiniGrid-Empty-Random-6x6-v0', n_envs=1, post_wrappers= [wrappers.RGBImgObsWrapper, wrappers.ImgObsWrapper])
# venv = VecTransposeImage(venv)


print(venv.reset())
print(venv.reset().shape)
base_ppo = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size= 10, n_steps= 100)

# logger.configure("cnn_empty_warmstart_bc_logs")
# bc_trainer = bc.BC(venv.observation_space, venv.action_space, is_image= False, expert_data=transitions, loss_type = "original", policy_class=policies.ActorCriticPolicy )
# # bc_trainer.policy = new_policy
# bc_trainer.train(n_epochs=50)
# bc_trainer.save_policy("cnn_empty_bc_warmstart.pt")


# base_ppo.policy = bc.reconstruct_policy("cnn_empty_bc_warmstart.pt")
logger.configure("empty_stack4_normal_gail")

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=10,
    gen_algo= base_ppo,
    n_disc_updates_per_round=1

)

gail_trainer.train(total_timesteps= int(10000))


gail_trainer.gen_algo.save("empty_stack4_gail_gen")
