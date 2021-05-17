from bac_utils.env_utils import minigrid_render, minigrid_get_env
import os, time
import numpy as np

import argparse

import matplotlib.pyplot as plt


import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc

import gym
import gym_minigrid


from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy


from imitation.algorithms import bc
from imitation.data.types import Trajectory


import dataclasses
import logging
import os
from typing import Callable, Dict, Iterable, Mapping, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.utils.data as th_data
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import on_policy_algorithm, preprocessing, vec_env

from imitation.data import buffer, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper, util


class BaC:
    def __init__(
        self,
        train_env,
        eval_env,
        bc_trainer,
        bac_classifier,
        expert_data,
        expert_batch_size: int = 32,
    ):

        self.train_env = train_env  # pass an instance of the environment
        self.eval_env = eval_env
        self.bc_trainer = (
            bc_trainer  # pass an instance of imitation.bc with a trained policy.
        )
        self.bac_classifier = bac_classifier

        self.expert_batch_size = expert_batch_size
        self.not_expert_dataset = []

        # taken from imitation.algorithms.adversarial
        self.expert_dataloader = self.trajectory_dataset_to_dataloader(
            expert_data, expert_batch_size
        )

    def trajectory_dataset_to_dataloader(self, dataset, batch_size):
        if isinstance(dataset, types.Transitions):
            if len(dataset) < batch_size:
                raise ValueError(
                    "Provided Transitions instance as `dataset` argument but "
                    "len(dataset) < batch_size. "
                    f"({len(dataset)} < {batch_size})."
                )

            self.data_loader = th_data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=types.transitions_collate_fn,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.data_loader = dataset
        return util.endless_iter(self.data_loader)

    def collect_not_expert(self):
        for traj in range(10):
            obs_list = []
            action_list = []
            obs = self.train_env.reset()
            obs_list.append(obs[0])

            for _ in range(50):
                action = self.train_env.action_space.sample()

                obs, _, done, _ = self.train_env.step(action)
                action_list.append(action[0])
                obs_list.append(obs[0])

                if done:
                    break

            self.not_expert_dataset.append(
                Trajectory(
                    obs=np.array(obs_list),
                    acts=np.array(action_list),
                    infos=np.array([{} for i in action_list]),
                )
            )

    def train_bac_classifier(self):
        # get not expert batches
        self.collect_not_expert()
        self.not_expert_data_loader = self.trajectory_dataset_to_dataloader(
            self.not_expert_dataset, self.expert_batch_size
        )

        print("not expert data collected")

    

    def make_train_batch(self) -> dict:

        expert_samples = dict(next(self.expert_dataloader))
        gen_samples = dict(next(self.not_expert_data_loader))

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)]
        )

        batch_dict = {
            "state": self._torchify_with_space(obs, self.discrim.observation_space),
            "action": self._torchify_with_space(acts, self.discrim.action_space),
            "next_state": self._torchify_with_space(
                next_obs, self.discrim.observation_space
            ),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
        }

        return batch_dict


