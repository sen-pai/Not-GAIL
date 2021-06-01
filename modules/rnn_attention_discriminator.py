import gym
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from imitation.util import networks
from imitation.data.types import Trajectory
from imitation.data import rollout

from stable_baselines3.common import preprocessing
from stable_baselines3.common.torch_layers import NatureCNN


class ActObsCRNNAttn(nn.Module):
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

        self.rnn = nn.Sequential(
            nn.LSTM(
                input_size=self.in_size,
                hidden_size=self.in_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )
        )

        self.attn = Attn(self.in_size)
        self.mlp = networks.build_mlp(
            **{
                "in_size": self.in_size,
                "out_size": 1,
                "hid_sizes": (32, 32),
                **mlp_kwargs,
            }
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

    def preprocess_trajectory(self, trajectory: Trajectory):
        """Pass a Trajectory dataclass instance """
        transitions = rollout.flatten_trajectories([trajectory])
        obs = self._torchify_with_space(transitions.obs, self.observation_space)
        acts = self._torchify_with_space(transitions.acts, self.action_space)

        return obs, acts

    def forward(self, trajectory, attention=False) -> th.Tensor:
        encoder_outputs, traj_embeddings = self.embedding(trajectory)
        batch_size = encoder_outputs.shape[0]
        if attention:
            attn_features = self.attn(encoder_outputs)
            return self.mlp(attn_features).view(batch_size)

        _, traj_embeddings = self.embedding(trajectory)
        return self.mlp(traj_embeddings).view(batch_size)

    # embeddings for triplet loss
    def embedding(self, trajectory, attention = True) -> th.Tensor:

        if isinstance(trajectory, list):
            all_cat = []

            for traj in trajectory:
                obs, acts = self.preprocess_trajectory(traj)

                obs_features = self.cnn_feature_extractor(obs)
                cat_inputs = th.cat((obs_features, acts), dim=1)
                all_cat.append(cat_inputs)

            lens = [x.shape[0] for x in all_cat]
            padded_cat = pad_sequence(all_cat, batch_first=True, padding_value=0)
            cat_rnn_inputs = pack_padded_sequence(
                padded_cat, lens, enforce_sorted=False, batch_first=True
            )
            encoder_outputs, (hidden_state, _) = self.rnn(cat_rnn_inputs)

            encoder_outputs = nn.utils.rnn.pad_packed_sequence(
                encoder_outputs, batch_first=True
            )[0]
            encoder_outputs = encoder_outputs[:, :, : int(self.in_size)]
            if attention:
                return self.attn(encoder_outputs), hidden_state
            return encoder_outputs, hidden_state
        else:
            obs, acts = self.preprocess_trajectory(trajectory)

            obs_features = self.cnn_feature_extractor(obs)
            cat_inputs = th.cat((obs_features, acts), dim=1)

            cat_rnn_inputs = cat_inputs.view(
                1, -1, self.in_size
            )  # batch, seq_len, feature_size
            encoder_outputs, (hidden_state, _) = self.rnn(cat_rnn_inputs)

            encoder_outputs = encoder_outputs[:, :, : int(self.in_size)]
            if attention:
                return self.attn(encoder_outputs), hidden_state

            return encoder_outputs, hidden_state

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 256), nn.ReLU(True), nn.Linear(256, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(
            encoder_outputs.view(-1, self.h_dim)
        )  # (b, s, h) -> (b * s, 1)
        attns = F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(
            2
        )  # (b*s, 1) -> (b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1)
        return feats

