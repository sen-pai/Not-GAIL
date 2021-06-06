import gym
import numpy as np
import torch as th
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from imitation.util import networks
from imitation.data.types import Trajectory
from imitation.data import rollout

from stable_baselines3.common import preprocessing
from stable_baselines3.common.torch_layers import NatureCNN

import torch.nn.functional as F

class CNNAutoEncoder(nn.Module):
    """CRNN that takes a trajectory of action, image observation pairs and returns a logit"""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.feature_dim = 256
        self.encoder = NatureCNN(observation_space, features_dim=self.feature_dim)
        

        self.flatten_decoder = nn.Sequential(
            
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.observation_space.shape[0], kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.device(), **kwargs)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor, space, normalize_images=True,
        )
        return preprocessed

    def forward(self, input) -> th.Tensor:
        # input = self._torchify_with_space(input, self.observation_space)
        # print(input.shape)
        encoder_vec = self.encoder(input)

        encoder_vec = th.reshape(encoder_vec, [-1, self.feature_dim, 1, 1])
        # print(encoder_vec.shape)
        return self.decoder(encoder_vec)


    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device

    def calc_loss(self, input):
        result = self.forward(input)
        return F.binary_cross_entropy(result, input)

    def generate(self, input):
        return self.forward(input)

    def encode(self, input: np.ndarray):
        input = self._torchify_array(input)
        input = input.permute(2,0,1)/255
        
        return np.array(self.encoder(th.unsqueeze(input,0))[0].detach())

    def decode(self, input: np.ndarray):
        input = th.reshape(self._torchify_array(input), [-1, self.feature_dim, 1, 1])

        return np.array(self.decoder(input)[0].detach())
