import gym
import numpy as np
import torch as th
from torch import nn
from imitation.util import networks
from imitation.data.types import Trajectory
from imitation.data import rollout

from stable_baselines3.common import preprocessing

import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size = 256):
        super(UnFlatten, self).__init__()
        self.size = size 
    
    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)

        self.feature_dim = 256
        self.encoder = NatureCNN(observation_space, features_dim=self.feature_dim)
        

class CNN_VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=256, z_dim=32):
        super(CNN_VAE, self).__init__()
        self.features_dim  = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(h_dim),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.observation_space.shape[0], kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    # def infer_h_dim(self, obs):
    #     self.h_dim = 

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = th.randn(*mu.size(), device=self.device())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print("H", h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, input) -> th.Tensor:
        # input = self._torchify_with_space(input, self.observation_space)
        # print(input.shape)
        encoder_vec = self.encoder(input)

        encoder_vec = th.reshape(encoder_vec, [-1, self.feature_dim, 1, 1])
        # print(encoder_vec.shape)
        return self.decoder(encoder_vec)

    def forward(self, x):
        z, _, _ = self.encode(x)
        return z

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
