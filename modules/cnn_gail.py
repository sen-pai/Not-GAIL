
import logging
from typing import Iterable, Mapping, Optional, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import on_policy_algorithm, vec_env

from imitation.data import types
from imitation.rewards import discrim_nets
from imitation.algorithms.adversarial import AdversarialTrainer
from imitation.rewards.discrim_nets import DiscrimNetGAIL

from .cnn_discriminator import ActObsCNN


class CNNGAIL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        discrim = None,
        *,
        # FIXME(sam) pass in discrim net directly; don't ask for kwargs indirectly
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning that accepts Image Obs

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `CNNGAIL` adds on top of its superclass initializer are
        as follows:

        Args:
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetGAIL.

        """
        discrim_kwargs = discrim_kwargs or {}

        if discrim == None:
            discrim = discrim_nets.DiscrimNetGAIL(
                venv.observation_space, venv.action_space, discrim_net=ActObsCNN, **discrim_kwargs
            )

        logging.info("using CNN GAIL")
        super().__init__(
            venv, gen_algo, discrim, expert_data, expert_batch_size, **kwargs
        )


class LogitDiscrimNetGAIL(DiscrimNetGAIL):
    """The discriminator to use for GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discrim_net: ActObsCNN,
        scale: bool = False,
    ):
        """Construct discriminator network.

        Args:
          observation_space: observation space for this environment.
          action_space: action space for this environment:
          discrim_net: a Torch module that takes an observation and action
            tensor as input, then computes the logits for GAIL.
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space, scale=scale
        )

        self.discriminator = discrim_net

        logging.info("using Logit GAIL")

    #override method
    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.logits_gen_is_high(state, action, next_state, done)
        rew = -(logits)
        assert rew.shape == state.shape[:1]
        return rew



"""
Example ways to call above classes
"""
if __name__ == "__main__":
    import gym 
    import gym_minigrid

    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")

    logit_discrim = LogitDiscrimNetGAIL(env.observation_space, env.action_space) 

    #pass an instance of logit_discrim, not the class
    gail_trainer = CNNGAIL(discrim=logit_discrim)
    # and so on..



"""
Original Discriminator for Referance
"""
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
#             # self.discriminator = ActObsMLP(
#             #     action_space=action_space,
#             #     observation_space=observation_space,
#             #     hid_sizes=(32, 32),
#             # )
#             self.discriminator = ObsMLP(
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
