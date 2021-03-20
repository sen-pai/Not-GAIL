# import abc
# import logging
# from typing import Optional

# import gym
# import numpy as np
# import torch as th
# import torch.nn.functional as F
# from stable_baselines3.common import preprocessing
# from torch import nn

# from imitation.rewards import common as rewards_common
# from imitation.rewards import reward_nets
# from imitation.util import networks


# class DiscrimNet(nn.Module, abc.ABC):
#     """Abstract base class for discriminator, used in AIRL and GAIL."""

#     def __init__(
#         self,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         scale: bool = False,
#     ):
#         super().__init__()
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.scale = scale

#     @abc.abstractmethod
#     def logits_gen_is_high(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#         log_policy_act_prob: Optional[th.Tensor] = None,
#     ) -> th.Tensor:
#         """Compute the discriminator's logits for each state-action sample.

#         A high value corresponds to predicting generator, and a low value corresponds to
#         predicting expert.

#         Args:
#             state: state at time t.
#             action: action taken at time t.
#             next_state: state at time t+1.
#             done: binary episode completion flag after action at time t.
#             log_policy_act_prob: log policy of novice taking `action`. This is
#                 only used for AIRL.

#         Returns:
#             disc_logits_gen_is_high: discriminator logits for a sigmoid
#                 activation. A high output indicates a generator-like transition.
#         """

#     def disc_loss(self, disc_logits_gen_is_high, labels_gen_is_one) -> th.Tensor:
#         """Compute discriminator loss.

#         Args:
#             disc_logits_gen_is_high: discriminator logits, as produced by
#                 `logits_gen_is_high`.
#             labels_gen_is_one: integer labels, with zero for expert and one for
#                 generator (novice).

#         Returns:
#             loss: scalar-valued discriminator loss."""
#         return F.binary_cross_entropy_with_logits(
#             disc_logits_gen_is_high, labels_gen_is_one.float()
#         )

#     def device(self) -> th.device:
#         """Heuristic to determine which device this module is on."""
#         first_param = next(self.parameters())
#         return first_param.device

#     @abc.abstractmethod
#     def reward_test(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         """Test-time reward for given states/actions."""

#     @abc.abstractmethod
#     def reward_train(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         """Train-time reward for given states/actions."""

#     @abc.abstractmethod
#     def neg_reward_test(
#         self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
#     ) -> th.Tensor:
#         """Test-time reward for given states/actions."""

#     @abc.abstractmethod
#     def neg_reward_train(
#         self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
#     ) -> th.Tensor:
#         """Train-time reward for given states/actions."""


#     def predict_reward_train(
#         self,
#         state: np.ndarray,
#         action: np.ndarray,
#         next_state: np.ndarray,
#         done: np.ndarray,
#     ) -> np.ndarray:
#         """Vectorized reward for training an imitation learning algorithm.

#         Args:
#             state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             action: The action input. Its shape is
#                 `(batch_size,) + action_space.shape`. The None dimension is
#                 expected to be the same as None dimension from `obs_input`.
#             next_state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             done: Whether the episode has terminated. Its shape is `(batch_size,)`.
#         Returns:
#             The rewards. Its shape is `(batch_size,)`.
#         """
#         return self._eval_reward(
#             is_train=True, state=state, action=action, next_state=next_state, done=done
#         )

#     def predict_reward_test(
#         self,
#         state: np.ndarray,
#         action: np.ndarray,
#         next_state: np.ndarray,
#         done: np.ndarray,
#     ) -> np.ndarray:
#         """Vectorized reward for training an expert during transfer learning.

#         Args:
#             state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             act: The action input. Its shape is
#                 `(batch_size,) + action_space.shape`. The None dimension is
#                 expected to be the same as None dimension from `obs_input`.
#             next_state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             done: Whether the episode has terminated. Its shape is `(batch_size,)`.
#         Returns:
#             The rewards. Its shape is `(batch_size,)`.
#         """
#         return self._eval_reward(
#             is_train=False, state=state, action=action, next_state=next_state, done=done
#         )

#     def _eval_reward(
#         self,
#         is_train: bool,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> np.ndarray:
#         (
#             state_th,
#             action_th,
#             next_state_th,
#             done_th,
#         ) = rewards_common.disc_rew_preprocess_inputs(
#             observation_space=self.observation_space,
#             action_space=self.action_space,
#             state=state,
#             action=action,
#             next_state=next_state,
#             done=done,
#             device=self.device(),
#             scale=self.scale,
#         )

#         with th.no_grad():
#             if is_train:
#                 rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
#             else:
#                 rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)

#         rew = rew_th.detach().cpu().numpy().flatten()
#         assert rew.shape == (len(state),)

#         return rew

#     #ADDED below 3 classes
#     def neg_predict_reward_train(
#         self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
#     ) -> np.ndarray:
#         """Vectorized reward for training an imitation learning algorithm.

#         Args:
#             state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             action: The action input. Its shape is
#                 `(batch_size,) + action_space.shape`. The None dimension is
#                 expected to be the same as None dimension from `obs_input`.
#             next_state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             done: Whether the episode has terminated. Its shape is `(batch_size,)`.
#         Returns:
#             The rewards. Its shape is `(batch_size,)`.
#         """
#         return self._eval_neg_reward(
#             is_train=True, state=state, action=action, next_state=next_state, done=done
#         )

#     def neg_predict_reward_test(
#         self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
#     ) -> np.ndarray:
#         """Vectorized reward for training an expert during transfer learning.

#         Args:
#             state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             act: The action input. Its shape is
#                 `(batch_size,) + action_space.shape`. The None dimension is
#                 expected to be the same as None dimension from `obs_input`.
#             next_state: The observation input. Its shape is
#                 `(batch_size,) + observation_space.shape`.
#             done: Whether the episode has terminated. Its shape is `(batch_size,)`.
#         Returns:
#             The rewards. Its shape is `(batch_size,)`.
#         """
#         return self._eval_neg_reward(
#             is_train=False, state=state, action=action, next_state=next_state, done=done
#         )

#     def _eval_neg_reward(
#         self,
#         is_train: bool,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> np.ndarray:
#         (state_th, action_th, next_state_th, done_th,) = rewards_common.disc_rew_preprocess_inputs(
#             observation_space=self.observation_space,
#             action_space=self.action_space,
#             state=state,
#             action=action,
#             next_state=next_state,
#             done=done,
#             device=self.device(),
#             scale=self.scale,
#         )

#         with th.no_grad():
#             if is_train:
#                 rew_th = self.neg_reward_train(state_th, action_th, next_state_th, done_th)
#             else:
#                 rew_th = self.neg_reward_test(state_th, action_th, next_state_th, done_th)


#         rew = rew_th.detach().cpu().numpy().flatten()
#         assert rew.shape == (len(state),)

#         return rew



# class ActObsMLP(nn.Module):
#     """Simple MLP that takes an action and observation and produces a single
#     output."""

#     def __init__(
#         self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
#     ):
#         super().__init__()

#         in_size = preprocessing.get_flattened_obs_dim(
#             observation_space
#         ) + preprocessing.get_flattened_obs_dim(action_space)
#         self.mlp = networks.build_mlp(
#             **{"in_size": in_size, "out_size": 1, **mlp_kwargs}
#         )

#     def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
#         cat_inputs = th.cat((obs, acts), dim=1)
#         outputs = self.mlp(cat_inputs)
#         return outputs.squeeze(1)


# class DiscrimNetGAIL(DiscrimNet):
#     """The discriminator to use for GAIL."""

#     def __init__(
#         self,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         discrim_net: Optional[nn.Module] = None,
#         scale: bool = False,
#     ):
#         """Construct discriminator network.

#         Args:
#           observation_space: observation space for this environment.
#           action_space: action space for this environment:
#           discrim_net: a Torch module that takes an observation and action
#             tensor as input, then computes the logits for GAIL.
#           scale: should inputs be rescaled according to declared observation
#             space bounds?
#         """
#         super().__init__(
#             observation_space=observation_space, action_space=action_space, scale=scale
#         )

#         if discrim_net is None:
#             self.discriminator = ActObsMLP(
#                 action_space=action_space,
#                 observation_space=observation_space,
#                 hid_sizes=(32, 32),
#             )
#         else:
#             self.discriminator = discrim_net

#         logging.info("using GAIL")

#     def logits_gen_is_high(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#         log_policy_act_prob: Optional[th.Tensor] = None,
#     ) -> th.Tensor:
#         """Compute the discriminator's logits for each state-action sample.

#         A high value corresponds to predicting generator, and a low value corresponds to
#         predicting expert.
#         """
#         logits = self.discriminator(state, action)
#         return logits

#     def reward_test(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         rew = self.reward_train(state, action, next_state, done)
#         assert rew.shape == state.shape[:1]
#         return rew

#     def reward_train(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         logits = self.logits_gen_is_high(state, action, next_state, done)
#         rew = -F.logsigmoid(logits)
#         assert rew.shape == state.shape[:1]
#         return rew




# class DiscrimNetNegGAIL(DiscrimNet):
#     """The discriminator to use for GAIL."""

#     def __init__(
#         self,
#         observation_space: gym.Space,
#         action_space: gym.Space,
#         discrim_net: Optional[nn.Module] = None,
#         scale: bool = False,
#     ):
#         """Construct discriminator network.

#         Args:
#           observation_space: observation space for this environment.
#           action_space: action space for this environment:
#           discrim_net: a Torch module that takes an observation and action
#             tensor as input, then computes the logits for GAIL.
#           scale: should inputs be rescaled according to declared observation
#             space bounds?
#         """
#         super().__init__(
#             observation_space=observation_space, action_space=action_space, scale=scale
#         )

#         if discrim_net is None:
#             self.discriminator = ActObsMLP(
#                 action_space=action_space,
#                 observation_space=observation_space,
#                 hid_sizes=(32, 32),
#             )
#         else:
#             self.discriminator = discrim_net

#         logging.info("using GAIL")

#     def logits_gen_is_high(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#         log_policy_act_prob: Optional[th.Tensor] = None,
#     ) -> th.Tensor:
#         """Compute the discriminator's logits for each state-action sample.

#         A high value corresponds to predicting generator, and a low value corresponds to
#         predicting expert.
#         """
#         logits = self.discriminator(state, action)
#         return logits

#     def reward_test(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         rew = self.reward_train(state, action, next_state, done)
#         assert rew.shape == state.shape[:1]
#         return rew

#     def reward_train(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         logits = self.logits_gen_is_high(state, action, next_state, done)
#         rew = -F.logsigmoid(logits)
#         assert rew.shape == state.shape[:1]
#         return rew
    
#     def neg_reward_test(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         rew = self.neg_reward_train(state, action, next_state, done)
#         assert rew.shape == state.shape[:1]
#         return rew

#     def neg_reward_train(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#     ) -> th.Tensor:
#         logits = self.logits_gen_is_high(state, action, next_state, done)
#         rew = -(-F.logsigmoid(logits))
#         assert rew.shape == state.shape[:1]
#         return rew
    

import abc
import logging
from typing import Optional

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.rewards import common as rewards_common
from imitation.rewards import reward_nets
from imitation.util import networks


class DiscrimNet(nn.Module, abc.ABC):
    """Abstract base class for discriminator, used in AIRL and GAIL."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, scale: bool = False,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.scale = scale

    @abc.abstractmethod
    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.

        Args:
            state: state at time t.
            action: action taken at time t.
            next_state: state at time t+1.
            done: binary episode completion flag after action at time t.
            log_policy_act_prob: log policy of novice taking `action`. This is
                only used for AIRL.

        Returns:
            disc_logits_gen_is_high: discriminator logits for a sigmoid
                activation. A high output indicates a generator-like transition.
        """

    def disc_loss(self, disc_logits_gen_is_high, labels_gen_is_one) -> th.Tensor:
        """Compute discriminator loss.

        Args:
            disc_logits_gen_is_high: discriminator logits, as produced by
                `logits_gen_is_high`.
            labels_gen_is_one: integer labels, with zero for expert and one for
                generator (novice).

        Returns:
            loss: scalar-valued discriminator loss."""
        return F.binary_cross_entropy_with_logits(
            disc_logits_gen_is_high, labels_gen_is_one.float()
        )

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device

    @abc.abstractmethod
    def reward_test(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        """Test-time reward for given states/actions."""

    @abc.abstractmethod
    def reward_train(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        """Train-time reward for given states/actions."""

    @abc.abstractmethod
    def neg_reward_test(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        """Test-time reward for given states/actions."""

    @abc.abstractmethod
    def neg_reward_train(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        """Train-time reward for given states/actions."""

    def predict_reward_train(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an imitation learning algorithm.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=True, state=state, action=action, next_state=next_state, done=done
        )

    def predict_reward_test(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an expert during transfer learning.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=False, state=state, action=action, next_state=next_state, done=done
        )

    def _eval_reward(
        self,
        is_train: bool,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> np.ndarray:
        (state_th, action_th, next_state_th, done_th,) = rewards_common.disc_rew_preprocess_inputs(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state=state,
            action=action,
            next_state=next_state,
            done=done,
            device=self.device(),
            scale=self.scale,
        )

        with th.no_grad():
            # if is_train:
            #     rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
            # else:
                # rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)
            if is_train:
                rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
            else:
                rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)


        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == (len(state),)

        return rew

    def neg_predict_reward_train(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an imitation learning algorithm.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_neg_reward(
            is_train=True, state=state, action=action, next_state=next_state, done=done
        )

    def neg_predict_reward_test(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an expert during transfer learning.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_neg_reward(
            is_train=False, state=state, action=action, next_state=next_state, done=done
        )

    def _eval_neg_reward(
        self,
        is_train: bool,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> np.ndarray:
        (state_th, action_th, next_state_th, done_th,) = rewards_common.disc_rew_preprocess_inputs(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state=state,
            action=action,
            next_state=next_state,
            done=done,
            device=self.device(),
            scale=self.scale,
        )

        with th.no_grad():
            if is_train:
                rew_th = self.neg_reward_train(state_th, action_th, next_state_th, done_th)
            else:
                rew_th = self.neg_reward_test(state_th, action_th, next_state_th, done_th)


        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == (len(state),)

        return rew


class ActObsMLP(nn.Module):
    """Simple MLP that takes an action and observation and produces a single
    output."""

    def __init__(self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs):
        super().__init__()

        in_size = preprocessing.get_flattened_obs_dim(
            observation_space
        ) + preprocessing.get_flattened_obs_dim(action_space)
        self.mlp = networks.build_mlp(**{"in_size": in_size, "out_size": 1, **mlp_kwargs})

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        cat_inputs = th.cat((obs, acts), dim=1)
        outputs = self.mlp(cat_inputs)
        return outputs.squeeze(1)

class DiscrimNetGAIL(DiscrimNet):
    """The discriminator to use for GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discrim_net: Optional[nn.Module] = None,
        scale: bool = False,
    ):
        """Construct discriminator network.

        Args:
          observation_space: observation space for this environment.
          action_space: action space for this environment:
          discrim_net: a Torch module that takes an observation and action
            tensor as input, then computes the logits for GAIL.
          scale: should inputs be rescaled according to declared observation
            space bounds?
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space, scale=scale
        )

        if discrim_net is None:
            self.discriminator = ActObsMLP(
                action_space=action_space, observation_space=observation_space, hid_sizes=(32, 32),
            )

        else:
            self.discriminator = discrim_net

        logging.info("using GAIL")

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.
        """
        logits = self.discriminator(state, action)
        return logits

    def reward_test(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        rew = self.reward_train(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def reward_train(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        logits = self.logits_gen_is_high(state, action, next_state, done)
        rew = -F.logsigmoid(logits)
        assert rew.shape == state.shape[:1]
        return rew

    def neg_reward_test(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        rew = self.neg_reward_train(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def neg_reward_train(
        self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> th.Tensor:
        logits = self.logits_gen_is_high(state, action, next_state, done)
        rew = -(-F.logsigmoid(logits)) #multiplied by -ve
        assert rew.shape == state.shape[:1]
        return rew

