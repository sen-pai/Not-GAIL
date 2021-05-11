import gym
import highway_env
import numpy as np

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

config = {
    "action": {"type": "DiscreteMetaAction"},
    "centering_position": [0.3, 0.5],
    "collision_reward": -1,
    "controlled_vehicles": 1,
    "duration": 100,
    "ego_spacing": 2,
    "initial_lane_id": None,
    "lanes_count": 2,
    "manual_control": False,
    # "observation": {
    #     "type": "GrayscaleObservation",
    #     "observation_shape": (128, 64),
    #     "stack_size": 1,
    #     "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
    #     "scaling": 1.75,
    # },
    'observation': {'type': 'Kinematics'},
    "offroad_terminal": True,
    "offscreen_rendering": True,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "policy_frequency": 1,
    "real_time_rendering": False,
    "render_agent": True,
    "reward_speed_range": [20, 30],
    "scaling": 5.5,
    "screen_height": 150,
    "screen_width": 600,
    "show_trajectories": False,
    "simulation_frequency": 15,
    "vehicles_count": 0,
    "vehicles_density": 1,
}


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make("highway-v0")
        env.configure(config)
        env.reset()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    model = PPO('MlpPolicy', env, verbose=1, n_steps = 256, tensorboard_log = 'mlp_logs/')
    model.learn(total_timesteps=int(1e5))

    model.save("mlp_ppo_highway_multi")

    