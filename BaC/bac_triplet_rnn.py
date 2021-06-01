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

from .bac_rnn import BaCRNN


class BaC_RNN_Triplet(BaCRNN):
    def __init__(
        self,
        train_env,
        bac_classifier,
        expert_data,
        bc_trainer=None,
        not_expert_data=None,
        nepochs: int = 10,
        batch_size: int = 10,
        triplet_epochs: int = 60,
        classify_epochs: int = 100,
    ):
        super().__init__(
            train_env,
            bac_classifier,
            expert_data,
            bc_trainer,
            not_expert_data,
            nepochs=nepochs,
            batch_size=batch_size,
        )

        self.bac_triplet_optimizer = th.optim.AdamW(
            self.bac_classifier.parameters(), lr=1e-4
        )
        # self.bac_triplet_optimizer = th.optim.AdamW(self.bac_classifier.parameters(), lr = 1e-4, weight_decay = 1e-3)

        self.bac_triplet_loss = nn.TripletMarginLoss(margin=1.0)

        self.collect_max = 200
        self.triplet_epochs = triplet_epochs
        self.classify_epochs = classify_epochs

        self.fixed_anchor = [
            next(self.expert_dataloader) for _ in range(self.batch_size)
        ]

    def bac_classifier_epoch(self):
        full_loss = 0
        self.bac_classifier.train()
        for j in range(10):

            batch = [next(self.expert_dataloader) for i in range(self.batch_size)]
            exp_labels = np.ones(self.batch_size, dtype=int)

            if j % 3 == 0:
                # if self.bc_trainer != None:
                batch.extend(
                    [
                        next(self.not_expert_from_bc_dataloader)
                        for i in range(self.batch_size)
                    ]
                )
            elif j % 3 == 1:
                batch.extend(
                    [next(self.not_expert_dataloader) for i in range(self.batch_size)]
                )
            else:
                batch.extend(
                    [
                        next(self.not_expert_from_expert_dataloader)
                        for i in range(self.batch_size)
                    ]
                )

            label = self._torchify_array(
                np.concatenate([exp_labels, np.zeros(self.batch_size, dtype=int),])
            )

            logits = self.bac_classifier(batch)
            loss = self.bac_loss(logits, label.float())
            self.bac_optimizer.zero_grad()
            loss.backward()
            self.bac_optimizer.step()
            full_loss += loss.data
        print(f"classifier loss {full_loss / 10}")

    def bac_triplet_epoch(self):
        self.bac_classifier.train()

        full_loss = 0
        for j in range(10):

            anchor = [next(self.expert_dataloader) for _ in range(self.batch_size)]
            # anchor = copy.deepcopy(self.fixed_anchor)
            positive = [next(self.expert_dataloader) for _ in range(self.batch_size)]
            if j % 3 == 0:
                # if self.bc_trainer != None:
                negative = [
                    next(self.not_expert_from_bc_dataloader)
                    for _ in range(self.batch_size)
                ]
            elif j % 3 == 1:
                negative = [
                    next(self.not_expert_dataloader) for _ in range(self.batch_size)
                ]
            else:
                negative = [
                    next(self.not_expert_from_expert_dataloader)
                    for _ in range(self.batch_size)
                ]

            anchor_hidden = self.bac_classifier.embedding(anchor)[1]
            positive_hidden = self.bac_classifier.embedding(positive)[1]
            negative_hidden = self.bac_classifier.embedding(negative)[1]

            triplet_loss = self.bac_triplet_loss(
                anchor_hidden, positive_hidden, negative_hidden
            )
            self.bac_triplet_optimizer.zero_grad()
            triplet_loss.backward()
            self.bac_triplet_optimizer.step()
            full_loss += triplet_loss.data
        print(f"triplet loss {full_loss / 10}")

    def train_bac_2halfs(self, filter=True):
        self.collect_not_expert_from_bc(filter=False)
        self.collect_not_expert(filter=False)
        self.collect_not_expert_from_expert(filter=False)

        for i in tqdm(range(self.triplet_epochs + self.classify_epochs)):

            # collect not expert after every 20 epochs
            if i % 20 == 0 and i > 1 and i > self.triplet_epochs:
                # self.bac_classifier_epoch()
                self.collect_not_expert_from_bc(filter)
                self.collect_not_expert(filter)
                self.collect_not_expert_from_expert(filter)

            # if i <= self.triplet_epochs:
            #     self.bac_triplet_epoch()
            # else:
            #     self.bac_classifier_epoch()

            if i <= self.triplet_epochs:
                self.bac_triplet_epoch()
            else:
                self.bac_classifier_epoch()

            if self.earlystopping():
                break

        print("bac training done")
