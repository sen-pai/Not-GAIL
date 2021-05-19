import gym
import numpy as np
import torch as th
from torch import nn

from imitation.util import networks
from imitation.data.types import Trajectory
from imitation.data import rollout

from stable_baselines3.common import preprocessing
from stable_baselines3.common.torch_layers import NatureCNN


class ActObsCRNN(nn.Module):
    """CRNN that takes a trajectory of action, image observation pairs and returns a logit"""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.cnn_feature_extractor = NatureCNN(observation_space, features_dim=512)
        
        self.in_size = (
            self.cnn_feature_extractor.features_dim
            + preprocessing.get_flattened_obs_dim(action_space)
        )
        
        self.rnn = nn.LSTM(
            input_size = self.in_size,
            hidden_size = self.in_size,
            num_layers = 1,
            bidirectional=False,
            batch_first=True,
        )

        
        self.mlp = networks.build_mlp(
            **{"in_size": self.in_size, "out_size": 1, "hid_sizes": (32, 32), **mlp_kwargs}
        )

    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.device(), **kwargs)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor, space, normalize_images=False,
        )
        return preprocessed
    
    def preprocess_trajectory(self, trajectory:Trajectory):
        """Pass a Trajectory dataclass instance """
        transitions = rollout.flatten_trajectories([trajectory])
        obs = self._torchify_with_space(transitions.obs, self.observation_space)
        acts = self._torchify_with_space(transitions.acts, self.action_space)
        # print(transitions.acts.shape)
        # print(transitions.obs.shape)
        return obs, acts


    def forward(self, trajectory:Trajectory) -> th.Tensor:
        obs, acts = self.preprocess_trajectory(trajectory)

        obs_features = self.cnn_feature_extractor(obs)
        # print(obs_features.shape)
        # print(acts.shape)
        cat_inputs = th.cat((obs_features, acts), dim=1)

        cat_rnn_inputs = cat_inputs.view(1, -1, self.in_size) #batch, seq_len, feature_size
        _, (hidden_state, _) = self.rnn(cat_rnn_inputs)

        outputs = self.mlp(hidden_state)
        return outputs.squeeze(1)

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device