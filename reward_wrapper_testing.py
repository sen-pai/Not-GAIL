import msvcrt
import numpy as np
from bac_utils.env_utils import minigrid_get_env
from BaC import bac_wrappers
import get_classifier



def cust_rew(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:

    return np.array([-1]*len(state))


venv = minigrid_get_env(
    'MiniGrid-MidEmpty-Random-6x6-v0',
    n_envs=1,
)


bac_trainer = get_classifier.get_classifier(venv)

venv = bac_wrappers.RewardVecEnvWrapperRNN(
    venv, 
    reward_fn=bac_trainer.predict, 
    bac_reward_flag=True   # Whether to use new_rews(False) or old_rews-new_rews(True)
)

x = ""
while x != "n":
    venv.render()
    x = msvcrt.getwch()

    if x == "a":
        action = [0]
    elif x == "d":
        action = [1]
    elif x == "n":
        break
    elif x == "w":
        action = [2]
    elif x == "e":
        action = [3]
    else:
        action = [int(x)]

    n_obs, reward, done, info = venv.step(action)
    
    print(action, reward)
    obs = n_obs
    if done[0]:
        print("done")
        obs = venv.reset()
    print()