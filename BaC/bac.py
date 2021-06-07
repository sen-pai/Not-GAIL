from utils.env_utils import minigrid_render, minigrid_get_env
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
import torch.nn as nn
import torch.utils.data as th_data
import torch.utils.tensorboard as thboard
from tqdm import tqdm
from stable_baselines3.common import on_policy_algorithm, preprocessing, vec_env

from imitation.data import buffer, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper, util
from imitation.data import rollout


class BaC:
    def __init__(
        self,
        train_env,
        eval_env,
        bc_trainer,
        bac_classifier,
        expert_data,
        expert_batch_size: int = 32,
        nepochs: int = 20,
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

        self.bac_optimizer = th.optim.Adam(self.bac_classifier.parameters())
        self.bac_loss = nn.BCEWithLogitsLoss()

        self.nepochs = nepochs
        print(f"BaC will be trained for {100*nepochs} epochs")


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

    def collect_not_expert(self, filter=False, cuttoff=0.7):
        self.not_expert_dataset = []
        for traj in range(10):
            obs_list = []
            action_list = []
            obs = self.train_env.reset()
            obs_list.append(obs[0])

            for _ in range(50):
                action = self.train_env.action_space.sample()

                obs, _, done, _ = self.train_env.step([action])
                action_list.append(action)

                if (
                    self.predict(
                        np.expand_dims(obs_list[-1], axis=0),
                        np.expand_dims(np.array(action_list[-1]), axis=0),
                    )
                    > cuttoff
                    and filter
                ):
                    # count += 1
                    del obs_list[-1]
                    del action_list[-1]

                obs_list.append(obs[0])

                if done:
                    break
            
            if action_list:
                self.not_expert_dataset.append(
                    Trajectory(
                        obs=np.array(obs_list),
                        acts=np.array(action_list),
                        infos=np.array([{} for i in action_list]),
                    )
                )
        self.not_expert_dataset = rollout.flatten_trajectories(self.not_expert_dataset)
        self.not_expert_dataloader = self.trajectory_dataset_to_dataloader(
            self.not_expert_dataset, self.expert_batch_size
        )

    def train_bac_classifier(self):
        self.bac_classifier.train()

        for i in tqdm(range(self.nepochs)):
            # filter = False if i ==0 else True
            filter = False
            
            self.collect_not_expert(filter=filter)
            for j in range(100):
                batch = self.make_train_batch()
                logits = self.bac_classifier(batch["state"], batch["action"])
                loss = self.bac_loss(logits, batch["labels_expert_is_one"].float())
                self.bac_optimizer.zero_grad()
                loss.backward()
                self.bac_optimizer.step()

            print(loss.data)

        print("bac training done")

    def save(self, save_path, save_name):
        os.chdir(save_path)
        th.save(self.bac_classifier.state_dict(),)

    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.bac_classifier.device(), **kwargs)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.bac_classifier.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor, space, normalize_images=False,
        )
        return preprocessed

    def predict(self, obs: np.array, act: np.array, return_logit=False):
        """
        predicts and returns either logit or prop
        """
        self.bac_classifier.eval()
        obs = self._torchify_with_space(obs, self.train_env.observation_space)
        act = self._torchify_with_space(act, self.train_env.action_space)

        logit = self.bac_classifier(obs, act)

        if return_logit:
            return logit
        else:
            probs = th.sigmoid(logit)  # no need for -logit as expert is 1
            return probs

    def make_train_batch(self) -> dict:

        expert_samples = dict(next(self.expert_dataloader))
        not_expert_samples = dict(next(self.not_expert_dataloader))

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [not_expert_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(not_expert_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], not_expert_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], not_expert_samples["acts"]])
        next_obs = np.concatenate(
            [expert_samples["next_obs"], not_expert_samples["next_obs"]]
        )
        dones = np.concatenate([expert_samples["dones"], not_expert_samples["dones"]])
        labels_expert_is_one = np.concatenate(
            [
                np.ones(self.expert_batch_size, dtype=int),
                np.zeros(self.expert_batch_size, dtype=int),
            ]
        )

        batch_dict = {
            "state": self._torchify_with_space(obs, self.train_env.observation_space),
            "action": self._torchify_with_space(acts, self.train_env.action_space),
            "next_state": self._torchify_with_space(
                next_obs, self.train_env.observation_space
            ),
            "done": self._torchify_array(dones),
            "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
        }

        return batch_dict

