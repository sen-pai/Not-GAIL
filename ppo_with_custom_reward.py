from stable_baselines3 import PPO
from stable_baselines3.common import policies

from utils.env_utils import minigrid_get_env
import numpy as np

from utils import env_wrappers

import utils.get_classifier as get_classifier

def cust_rew(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:

    return np.array([-1]*len(state))


venv = minigrid_get_env('MiniGrid-MidEmpty-Random-6x6-v0',n_envs = 1)
# util.make_vec_env(
#     'MiniGrid-Empty-Random-6x6-v0',
#     n_envs=1,
#     post_wrappers=[wrap.FlatObsWrapper],
#     post_wrappers_kwargs=[{}],
# )


# venv_buffering = wrappers.BufferingWrapper(venv)
# venv_wrapped = vec_env.VecNormalize(
#     venv_buffering,
#     norm_reward=False,
#     norm_obs=False,
# )

# with open("discrims/gail_discrim9.pkl", "rb") as f:
#     discrim = pickle.load(f)

bac_trainer = get_classifier.get_classifier(venv)

venv = env_wrappers.RewardVecEnvWrapperRNN(
    venv, 
    reward_fn=bac_trainer.predict, 
    bac_reward_flag=True   # Whether to use new_rews(False) or old_rews-new_rews(True)
)

# venv = bac_wrappers.RewardVecEnvWrapper(
#     venv, 
#     reward_fn=cust_rew, 
#     bac_reward_flag=False   # Whether to use new_rews(False) or old_rews-new_rews(True)
# )


model = PPO(policies.ActorCriticCnnPolicy, venv, verbose=1, batch_size=50, n_steps=50)#PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps= int(5e4))#, callback=eval_callback)

model.save("models/ppo_empty_normal")