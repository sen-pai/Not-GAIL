from stable_baselines3 import PPO
from stable_baselines3.common import policies
from imitation.util import util
from utils.env_utils import minigrid_get_env
import numpy as np

from utils import env_wrappers
import utils.get_classifier as get_classifier
import gym_custom

def cust_rew(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:

    return np.array([-1]*len(state))


# venv = minigrid_get_env('MiniGrid-MidEmpty-Random-6x6-v0',n_envs = 1)
venv = util.make_vec_env(
    'CoverAllTargetsDiscrete-v0',
    n_envs=1
    # post_wrappers=[wrap.FlatObsWrapper],
    # post_wrappers_kwargs=[{}],
)


# with open("discrims/gail_discrim9.pkl", "rb") as f:
#     discrim = pickle.load(f)

# bac_trainer = get_classifier.get_classifier(venv)

# venv = env_wrappers.RewardVecEnvWrapperRNN(
#     venv, 
#     reward_fn=bac_trainer.predict, 
#     bac_reward_flag=True   # Whether to use new_rews(False) or old_rews-new_rews(True)
# )

# venv = bac_wrappers.RewardVecEnvWrapper(
#     venv, 
#     reward_fn=cust_rew, 
#     bac_reward_flag=False   # Whether to use new_rews(False) or old_rews-new_rews(True)
# )


model = PPO(policies.ActorCriticPolicy, venv, verbose=1, batch_size=100, n_steps=200)#PPO('MlpPolicy', env, verbose=1)
# model.load("models/ppo_cover_all_targets")
model.learn(total_timesteps= int(1e5))#, callback=eval_callback)

model.save("models/ppo_cover_all_targets")