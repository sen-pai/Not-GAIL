import os
from imitation.data.types import Trajectory
import pickle

import time
import argparse
import numpy as np
import gym
import highway_env


traj_dataset = []
obs_list = []
action_list = []
info_list = []


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
    'observation': {'type': 'Kinematics'},
    "offroad_terminal": True,
    "offscreen_rendering": False,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "policy_frequency": 1,
    "real_time_rendering": False,
    "render_agent": True,
    "reward_speed_range": [20, 30],
    "scaling": 5.5,
    "screen_height": 150,
    "screen_width": 600,
    "show_trajectories": True,
    "simulation_frequency": 15,
    "vehicles_count": 2,
    "vehicles_density": 1,
}

env = gym.make("highway-v0")
# env = gym.make("roundabout-v0")

env.configure(config)
env.reset()

traj_dataset = []
for traj in range(30):
    obs_list = []
    action_list = []
    obs = env.reset()
    print(obs, traj)
    obs_list.append(obs)
    for i in range(80):
        action = 0
        obs, reward, done, info = env.step(action)
        print(action)
        action_list.append(action)
        obs_list.append(obs)
        env.render()
        if done:
            break
    traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array([{} for i in action_list])))


with open('collision_highway_traj.pkl', 'wb') as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
