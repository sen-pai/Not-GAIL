from utils.env_utils import minigrid_render, minigrid_get_env
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
        bac_classifier,
        expert_data,
        bc_trainer=None,
        not_expert_data=None,
        nepochs: int = 10,
        batch_size: int = 10,
    ):

        self.train_env = train_env  # pass an instance of the environment
        self.bc_trainer = (
            bc_trainer  # pass an instance of imitation.bc with a trained policy.
        )
        self.bac_classifier = bac_classifier

        self.not_expert_data = not_expert_data
        self.expert_data = expert_data

        # taken from imitation.algorithms.adversarial
        self.expert_dataloader = util.endless_iter(expert_data)

        self.provided_not_expert_dataloader = None
        if not_expert_data:
            self.provided_not_expert_dataloader = util.endless_iter(not_expert_data)

        self.bac_optimizer = th.optim.AdamW(self.bac_classifier.parameters())
        
        self.bac_loss = nn.BCEWithLogitsLoss()
        
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.collect_max = 150

    def train_bac_classifier(self):
        if self.bc_trainer != None:
            self.collect_not_expert_from_bc(filter = False)
        self.collect_not_expert(filter = False)
        self.collect_not_expert_from_expert(filter = False)

        for i in tqdm(range(self.nepochs)):
            full_loss = 0

            if i%25 == 0 and i > 20:
                if self.bc_trainer != None:
                    self.collect_not_expert_from_bc(filter = True)
                self.collect_not_expert(filter = True)
                self.collect_not_expert_from_expert(filter = True)
                        
            self.bac_classifier.train()
            for j in range(10):

                batch = [next(self.expert_dataloader) for i in range(self.batch_size)]
                exp_labels = np.ones(self.batch_size, dtype=int)

                if i % 2 == 0:
                    if self.bc_trainer != None:
                        batch.extend([next(self.not_expert_from_bc_dataloader) for i in range(self.batch_size)])
                    else:
                        batch.extend([next(self.not_expert_dataloader) for i in range(self.batch_size)])
                else:
                    batch.extend([next(self.not_expert_from_expert_dataloader) for i in range(self.batch_size)])
                    

                label = self._torchify_array(
                    np.concatenate([exp_labels, np.zeros(self.batch_size, dtype=int),])
                )

                logits = self.bac_classifier(batch)
                loss = self.bac_loss(logits, label.float())
                self.bac_optimizer.zero_grad()
                loss.backward()
                self.bac_optimizer.step()
                full_loss += loss.data
            print(full_loss / 10)

            if self.earlystopping():
                break

        print("bac training done")

    def expert_warmstart(self):
        self.bac_classifier.train()

        print(f"warm start for {20} epochs with batch size {5}")
        for i in tqdm(range(10)):
            full_loss = 0
            for j in range(2):
                batch = [next(self.expert_dataloader) for i in range(5)]

                label = self._torchify_array(np.ones(5, dtype=int))

                logits = self.bac_classifier(batch)
                loss = self.bac_loss(logits, label.float())
                self.bac_optimizer.zero_grad()
                loss.backward()
                self.bac_optimizer.step()
                full_loss += loss.item()

            print("expert only:", full_loss / 10)

        expert_probs_avg = 0
        for traj in self.expert_data:
            expert_probs_avg += self.predict(traj).item()

        print(f"expert probs sanity check {expert_probs_avg/len(self.expert_data)}")
        print("bac warmstart done")


    def save(self, save_path, save_name):
        os.chdir(save_path)
        th.save(self.bac_classifier.state_dict(),save_name)

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
        for _ in range(self.collect_max):
            obs_list = []
            action_list = []
            obs = self.train_env.reset()
            obs_list.append(obs[0])

            for i in range(4):
                # action = self.train_env.action_space.sample()
                action = random.sample([0, 1, 2], 1)[0]
                # print(action)

                obs, _, done, _ = self.train_env.step([action])
                action_list.append(action)
                obs_list.append(obs[0])

                if done:
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
        self.not_expert_dataloader = util.endless_iter(self.not_expert_dataset)
        
    def collect_not_expert_from_bc(self, filter = False, cutoff = 0.9):
        assert self.bc_trainer != None, "Need a trained BC" 
        self.not_expert_from_bc_dataset = []
        for _ in range(self.collect_max):
            obs_list = []
            action_list = []
            ok_flag = True
            obs = self.train_env.reset()
            obs_list.append(obs[0])

            #bc rollout
            for j in range(random.sample(list(range(1)), 1)[0]):
                action, _ = self.bc_trainer.policy.predict(obs, deterministic=True)
                obs, _, done, _ = self.train_env.step(action)
                action_list.append(action[0])
                obs_list.append(obs[0])
                if done:
                    ok_flag = False
                    break
            
            #continue with random actions
            for i in range(2):
                if not ok_flag:
                    break

                action = random.sample([0, 1, 2], 1)[0]
                obs, _, done, _ = self.train_env.step([action])
                action_list.append(action)
                obs_list.append(obs[0])

                if done:
                    ok_flag = False
                    break

            if len(action_list) >= 1 and ok_flag:
                collected_traj = Trajectory(
                        obs=np.array(obs_list),
                        acts=np.array(action_list),
                        infos=np.array([{} for i in action_list]),
                    )
                if filter:
                    if self.predict(collected_traj) < cutoff:
                        self.not_expert_from_bc_dataset.append(collected_traj)
                else:
                    self.not_expert_from_bc_dataset.append(collected_traj)

        print(f"not expert from bc dataset size: {len(self.not_expert_from_bc_dataset)}")

        self.not_expert_from_bc_dataloader = util.endless_iter(self.not_expert_from_bc_dataset)


    def collect_not_expert_from_expert(self, filter = False, cutoff = 0.9):
        self.not_expert_from_expert_dataset = []

        for _ in range(int(self.collect_max)):
            expert_traj = copy.deepcopy(next(self.expert_dataloader))
            obs_list = expert_traj.obs.tolist()
            act_list = expert_traj.acts.tolist()

            if len(act_list) < 5:
                continue

            for _ in range(random.sample(list(range(3)),1)[0]):
                del obs_list[-1]
                del act_list[-1]
                if len(act_list) < 2:
                    break


            if len(act_list) >= 1:
                collected_traj = Trajectory(
                        obs=np.array(obs_list),
                        acts=np.array(act_list),
                        infos=np.array([{} for i in act_list]),
                    )
                
                if filter:
                    if self.predict(collected_traj) < cutoff:
                        self.not_expert_from_expert_dataset.append(collected_traj)
                else:
                    self.not_expert_from_expert_dataset.append(collected_traj)

        # for _ in range(int(self.collect_max/2)):
        #     expert_traj = copy.deepcopy(next(self.expert_dataloader))
        #     obs_list = expert_traj.obs.tolist()
        #     act_list = expert_traj.acts.tolist()

        #     if len(act_list) < 5:
        #         continue
            


        #     if len(act_list) >= 1:
        #         collected_traj = Trajectory(
        #                 obs=np.array(obs_list),
        #                 acts=np.array(act_list),
        #                 infos=np.array([{} for i in act_list]),
        #             )
                
        #         if filter:
        #             if self.predict(collected_traj) < cutoff:
        #                 self.not_expert_from_expert_dataset.append(collected_traj)
        #         else:
        #             self.not_expert_from_expert_dataset.append(collected_traj)
        
        print(f"not expert from expert dataset size: {len(self.not_expert_from_expert_dataset)}")

        self.not_expert_from_expert_dataloader = util.endless_iter(self.not_expert_from_expert_dataset)


    def earlystopping(self):
        expert_probs_avg = 0
        for traj in self.expert_data:
            expert_probs_avg += self.predict(traj).item()

        expert_probs_avg /= len(self.expert_data)
        not_expert_probs_avg = 0

        if self.bc_trainer != None:
            for i, traj in enumerate(self.not_expert_from_bc_dataset):
                not_expert_probs_avg += self.predict(traj).item()
                if i > 100:
                    break
            total_not_expert = len(self.not_expert_from_bc_dataset)
        else:
            for i, traj in enumerate(self.not_expert_dataset):
                not_expert_probs_avg += self.predict(traj).item()
                if i > 100:
                    break
            total_not_expert = len(self.not_expert_dataset)

        for i, traj in enumerate(self.not_expert_from_expert_dataset):
            not_expert_probs_avg += self.predict(traj).item()
            if i > 100:
                break

        not_expert_probs_avg /= total_not_expert + len(
            self.not_expert_from_expert_dataset
        )

        print(
            f"earlystopping: expert = {expert_probs_avg} not expert = {not_expert_probs_avg}"
        )
        # if expert_probs_avg > 0.95 and not_expert_probs_avg < 0.05:
        #     return True
        return False
