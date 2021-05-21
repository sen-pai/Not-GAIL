from bac_utils.env_utils import minigrid_render, minigrid_get_env
import os, time
import numpy as np

import argparse
import random
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

import itertools
import copy


class BaCRNN:
    def __init__(
        self,
        train_env,
        bc_trainer,
        bac_classifier,
        expert_data,
        eval_env=None,
        not_expert_data=None,
        nepochs: int = 10,
    ):

        self.train_env = train_env  # pass an instance of the environment
        self.eval_env = eval_env
        self.bc_trainer = (
            bc_trainer  # pass an instance of imitation.bc with a trained policy.
        )
        self.bac_classifier = bac_classifier

        self.not_expert_data = not_expert_data
        self.expert_data = expert_data

        # taken from imitation.algorithms.adversarial
        self.expert_dataloader = util.endless_iter(expert_data)

        self.not_expert_dataloader = None
        if not_expert_data:
            self.not_expert_dataloader = util.endless_iter(not_expert_data)

        self.bac_optimizer = th.optim.Adam(self.bac_classifier.parameters())
        self.bac_loss = nn.BCEWithLogitsLoss()

        self.nepochs = nepochs
        print(f"BaC will be trained for {100*nepochs} epochs")

    def train_bac_classifier(self):

        for i in tqdm(range(self.nepochs)):
            full_loss = 0

            if i %2 == 1:
                self.collect_not_expert(filter = True)
            else:
                self.collect_not_expert_from_expert(filter = True)

            # self.collect_not_expert()
            self.bac_classifier.train()
            for j in range(2):
                batch = [next(self.expert_dataloader) for i in range(5)]
                batch.extend([next(self.not_expert_dataloader) for i in range(5)])

                label = self._torchify_array(
                    np.concatenate([np.ones(5, dtype=int), np.zeros(5, dtype=int),])
                )

                logits = self.bac_classifier(batch)
                loss = self.bac_loss(logits, label.float())
                self.bac_optimizer.zero_grad()
                loss.backward()
                self.bac_optimizer.step()
                full_loss += loss.data
            print(full_loss / 20)

        print("bac training done")

    def expert_warmstart(self):
        self.bac_classifier.train()

        print(f"warm start for {1000} epochs with batch size {5}")
        for i in tqdm(range(10)):
            full_loss = 0
            for j in range(10):
                batch = [next(self.expert_dataloader) for i in range(5)]

                label = self._torchify_array(np.ones(5, dtype=int))

                logits = self.bac_classifier(batch)
                loss = self.bac_loss(logits, label.float())
                self.bac_optimizer.zero_grad()
                loss.backward()
                self.bac_optimizer.step()
                full_loss += loss.data

            print("expert only:", full_loss / 10)

        expert_probs_avg = 0
        for traj in self.expert_data:
            expert_probs_avg += self.predict(traj)

        print(f"expert probs sanity check {expert_probs_avg/len(self.expert_data)}")

        print("bac warmstart done")


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

    def predict(self, traj, return_logit=False):
        """
        predicts and returns either logit or prop
        """
        self.bac_classifier.eval()

        logit = self.bac_classifier(traj)

        if return_logit:
            return logit
        else:
            probs = th.sigmoid(logit)  # no need for -logit as expert is 1
            return probs

    def collect_not_expert(self, filter = False, cutoff = 0.9):
        self.not_expert_dataset = []
        for _ in range(50):
            obs_list = []
            action_list = []
            obs = self.train_env.reset()
            obs_list.append(obs[0])

            for i in range(50):
                # action = self.train_env.action_space.sample()
                action = random.sample([0, 1, 2], 1)[0]
                # print(action)

                obs, _, done, _ = self.train_env.step([action])
                action_list.append(action)
                obs_list.append(obs[0])

                if done:
                    # print("done")
                    break

            if len(action_list) >= 1:
                collected_traj = Trajectory(
                        obs=np.array(obs_list),
                        acts=np.array(action_list),
                        infos=np.array([{} for i in action_list]),
                    )
                if filter:
                    if self.predict(collected_traj) < cutoff:
                        self.not_expert_dataset.append(collected_traj)
                else:
                    self.not_expert_dataset.append(collected_traj)

        print(f"not expert dataset size: {len(self.not_expert_dataset)}")
        # if len(self.not_expert_dataset) > 5:

        # self.not_expert_dataset = rollout.flatten_trajectories(self.not_expert_dataset)
        self.not_expert_dataloader = util.endless_iter(self.not_expert_dataset)
        # print("done func")
        # else:
        #     print("sadasdasdsadasds")
        #     self.not_expert_dataloader = util.endless_iter(self.not_expert_data)

    def collect_not_expert_from_expert(self, filter = False, cutoff = 0.9):
        self.not_expert_dataset = []

        for _ in range(30):
            expert_traj = copy.deepcopy(next(self.expert_dataloader))
            obs_list = expert_traj.obs.tolist()
            act_list = expert_traj.acts.tolist()

            if len(act_list) < 5:
                continue

            for _ in range(random.sample([1, 2, 3, 4],1)[0]):
                del obs_list[-1]
                del act_list[-1]

            collected_traj = Trajectory(
                    obs=np.array(obs_list),
                    acts=np.array(act_list),
                    infos=np.array([{} for i in act_list]),
                )
            
            if filter:
                if self.predict(collected_traj) < cutoff:
                    self.not_expert_dataset.append(collected_traj)
                else:
                    self.not_expert_dataset.append(collected_traj)
        
        print(f"not expert from expert dataset size: {len(self.not_expert_dataset)}")

        self.not_expert_dataloader = util.endless_iter(self.not_expert_dataset)

