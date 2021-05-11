import gym
import highway_env
import numpy as np
from stable_baselines3 import PPO

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
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 1,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    # 'observation': {'type': 'Kinematics'},
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
    "show_trajectories": False,
    "simulation_frequency": 15,
    "vehicles_count": 10,
    "vehicles_density": 1,
}

env = gym.make("highway-v0")
# env = gym.make("roundabout-v0")

env.configure(config)
env.reset()


# Load saved model
model = PPO.load("ppo_highway")
# model = PPO.load("mlp_ppo_highway_multi_w_v")


obs = env.reset()
env.render()
print("rendered")
# Evaluate the agent
episode_reward = 0
for _ in range(10):
    for i, _ in enumerate(range(1000)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done or info.get("is_success", False):
            print("Reward:", episode_reward, "Success?", info.get("is_success", False))
            print(i)
            episode_reward = 0.0
            obs = env.reset()
            break