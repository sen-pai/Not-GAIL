from torch.utils.data import dataloader
from utils.env_utils import minigrid_render, minigrid_get_env
import os, time
import numpy as np


import pickle5 as pickle
from imitation.data import rollout
from imitation.util import logger, util
from imitation.algorithms import bc

import gym
import gym_minigrid


import dataclasses
import logging
import os
from typing import Callable, Dict, Iterable, Mapping, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from stable_baselines3.common import on_policy_algorithm, preprocessing, vec_env

from imitation.rewards import common as rew_common
from imitation.util import logger, reward_wrapper, util
from imitation.data import rollout

import itertools
import copy


class VAE_trainer:
    def __init__(
        self, train_env, vae_class, nepochs: int = 25,
    ):
        self.train_env = train_env  # pass an instance of the environment
        self.vae = vae_class
        self.optimizer = th.optim.AdamW(self.vae.parameters())
        self.scheduler = th.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones =[15, 30, 50])
        self.batch_size = 50
        self.nepochs = nepochs

    def collect_random_obs(self, buffer_size=1000):
        self.buffer = []
        self.buffer.append(self.train_env.reset()[0])
        for i in range(buffer_size):
            action = self.train_env.action_space.sample()
            obs, _, done, _ = self.train_env.step([action])
            self.buffer.append(obs[0])
            if done:
                self.buffer.append(self.train_env.reset()[0])

    def loss_fn(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        BCE = F.mse_loss(recon_x, x, size_average=False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * th.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD

    def train(self):
        for epoch in tqdm(range(self.nepochs)):
            self.collect_random_obs()
            dataloader = util.endless_iter(self.buffer)
            for k in range(10):
                obs = [next(dataloader) for _ in range(self.batch_size)]
                obs = self._torchify_with_space(obs, self.train_env.observation_space)
                recon_obs, mu, logvar = self.vae.reconstruct(obs)
                loss, bce, kld = self.loss_fn(recon_obs, obs, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(
                    epoch + 1,
                    self.nepochs,
                    loss.item() / self.batch_size,
                    bce.item() / self.batch_size,
                    kld.item() / self.batch_size,
                ))
            
            self.scheduler.step()

    def save(self, save_path, save_name):
        os.chdir(save_path)
        th.save(self.vae.state_dict(), save_name)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.vae.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor, space, normalize_images=True,
        )
        return preprocessed