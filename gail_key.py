import copy
import os

import gym
import gym_minigrid
import numpy as np
import pickle5 as pickle
import torch as th
from gym_minigrid import wrappers
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger, util
from stable_baselines3 import PPO
from stable_baselines3.common import policies
import msvcrt


with open("traj_datasets/flat_empty_6x6_traj.pkl", "rb") as f:
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)


venv = util.make_vec_env(
    "MiniGrid-Empty-6x6-v0",
    n_envs=1,
    post_wrappers=[wrappers.FlatObsWrapper],
    post_wrappers_kwargs=[{}],
)


base_ppo = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size=50, n_steps=50)

logger.configure("MiniGrid-Empty-6x6-v0")

gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=60,
    gen_algo=base_ppo,
    n_disc_updates_per_round=1,
    normalize_reward=False,
    normalize_obs=False
)

total_timesteps = 8000
gail_trainer.train(total_timesteps=total_timesteps)
    # gail_trainer.gen_algo.save("gens/gail_gen_"+str(i))

    # with open('discrims/gail_discrim'+str(i)+'.pkl', 'wb') as handle:
    #     pickle.dump(gail_trainer.discrim, handle, protocol=pickle.HIGHEST_PROTOCOL)


discrim = gail_trainer.discrim
x = ""
obs = venv.reset()
while x != "n":
    venv.render()
    x = msvcrt.getwch()

    if x == "a":
        action = [0]
    elif x == "d":
        action = [1]
    elif x == "n":
        break
    elif x == "w":
        action = [2]
    elif x == "e":
        action = [3]
    else:
        action = [int(x)]

    n_obs, reward, done, info = venv.step(action)

    rew =th.sigmoid(-th.tensor(discrim.predict_reward_test(state = obs, action = action, next_state = n_obs, done = done)))

    print(action, rew)
    obs = n_obs
    if done:
        print("done")
        obs = venv.reset()
        tot_rew = 0


