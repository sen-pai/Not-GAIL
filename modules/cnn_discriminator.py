import gym
import numpy as np
import torch as th
from torch import nn

from imitation.util import networks

from stable_baselines3.common import preprocessing
from stable_baselines3.common.torch_layers import NatureCNN


class ActObsCNN(nn.Module):
    """CNN that takes an action and an image observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        self.cnn_feature_extractor = NatureCNN(observation_space, features_dim=512)

        in_size = (
            self.cnn_feature_extractor.features_dim
            + preprocessing.get_flattened_obs_dim(action_space)
        )
        self.mlp = networks.build_mlp(
            **{"in_size": in_size, "out_size": 1, "hid_sizes": (32, 32), **mlp_kwargs}
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        obs_features = self.cnn_feature_extractor(obs)
        cat_inputs = th.cat((obs_features, acts), dim=1)
        outputs = self.mlp(cat_inputs)
        return outputs.squeeze(1)

    # embeddings for triplet loss
    # def embedding(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:

    #         for traj in trajectory:
    #             obs, acts = self.preprocess_trajectory(traj)

    #             obs_features = self.cnn_feature_extractor(obs)
    #             cat_inputs = th.cat((obs_features, acts), dim=1)
    #             all_cat.append(cat_inputs)

    #         lens = [x.shape[0] for x in all_cat]
    #         padded_cat = pad_sequence(all_cat, batch_first=True, padding_value=0)
    #         cat_rnn_inputs = pack_padded_sequence(
    #             padded_cat, lens, enforce_sorted=False, batch_first=True
    #         )
    #         encoder_outputs, (hidden_state, _) = self.rnn(cat_rnn_inputs)

    #         encoder_outputs = nn.utils.rnn.pad_packed_sequence(
    #             encoder_outputs, batch_first=True
    #         )[0]
    #         encoder_outputs = encoder_outputs[:, :, : int(self.in_size)]
    #         if attention:
    #             return self.attn(encoder_outputs), self.bound_hidden(hidden_state)
    #         return encoder_outputs, self.bound_hidden(hidden_state)
    #     else:
    #         obs, acts = self.preprocess_trajectory(trajectory)

    #         obs_features = self.cnn_feature_extractor(obs)
    #         cat_inputs = th.cat((obs_features, acts), dim=1)

    #         cat_rnn_inputs = cat_inputs.view(
    #             1, -1, self.in_size
    #         )  # batch, seq_len, feature_size
    #         encoder_outputs, (hidden_state, _) = self.rnn(cat_rnn_inputs)

    #         encoder_outputs = encoder_outputs[:, :, : int(self.in_size)]
    #         if attention:
    #             return self.attn(encoder_outputs), self.bound_hidden(hidden_state)

    #         return encoder_outputs, self.bound_hidden(hidden_state)


    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device


class ObsCNN(nn.Module):
    """CNN that takes an action and an image observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        self.cnn_feature_extractor = NatureCNN(observation_space, features_dim=512)

        in_size = (
            self.cnn_feature_extractor.features_dim
        )
        self.mlp = networks.build_mlp(
            **{"in_size": in_size, "out_size": 1, "hid_sizes": (32, 32), **mlp_kwargs}
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        obs_features = self.cnn_feature_extractor(obs)
        outputs = self.mlp(obs_features)
        return outputs.squeeze(1)

    # embeddings for triplet loss
    # def embedding(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:

    #         for traj in trajectory:
    #             obs, acts = self.preprocess_trajectory(traj)

    #             obs_features = self.cnn_feature_extractor(obs)
    #             cat_inputs = th.cat((obs_features, acts), dim=1)
    #             all_cat.append(cat_inputs)

    #         lens = [x.shape[0] for x in all_cat]
    #         padded_cat = pad_sequence(all_cat, batch_first=True, padding_value=0)
    #         cat_rnn_inputs = pack_padded_sequence(
    #             padded_cat, lens, enforce_sorted=False, batch_first=True
    #         )
    #         encoder_outputs, (hidden_state, _) = self.rnn(cat_rnn_inputs)

    #         encoder_outputs = nn.utils.rnn.pad_packed_sequence(
    #             encoder_outputs, batch_first=True
    #         )[0]
    #         encoder_outputs = encoder_outputs[:, :, : int(self.in_size)]
    #         if attention:
    #             return self.attn(encoder_outputs), self.bound_hidden(hidden_state)
    #         return encoder_outputs, self.bound_hidden(hidden_state)
    #     else:
    #         obs, acts = self.preprocess_trajectory(trajectory)

    #         obs_features = self.cnn_feature_extractor(obs)
    #         cat_inputs = th.cat((obs_features, acts), dim=1)

    #         cat_rnn_inputs = cat_inputs.view(
    #             1, -1, self.in_size
    #         )  # batch, seq_len, feature_size
    #         encoder_outputs, (hidden_state, _) = self.rnn(cat_rnn_inputs)

    #         encoder_outputs = encoder_outputs[:, :, : int(self.in_size)]
    #         if attention:
    #             return self.attn(encoder_outputs), self.bound_hidden(hidden_state)

    #         return encoder_outputs, self.bound_hidden(hidden_state)


    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device