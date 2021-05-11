import gym
import highway_env
import numpy as np

from stable_baselines3 import PPO

config = {
    "action": {"type": "DiscreteMetaAction"},
    "centering_position": [0.3, 0.5],
    "collision_reward": -1,
    "controlled_vehicles": 1,
    "duration": 40,
    "ego_spacing": 2,
    "initial_lane_id": None,
    "lanes_count": 2,
    "manual_control": False,
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    #  'observation': {'type': 'Kinematics'},
    "offroad_terminal": False,
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
    "vehicles_count": 50,
    "vehicles_density": 1,
}

env = gym.make("highway-v0")
env.configure(config)
env.reset()


model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=int(1e5))

model.save("ppo_highway")

# Load saved model
# model = PPO.load("ppo_highway", env=env)

# obs = env.reset()

# # Evaluate the agent
# episode_reward = 0
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     episode_reward += reward
#     if done or info.get("is_success", False):
#         print("Reward:", episode_reward, "Success?", info.get("is_success", False))
#         episode_reward = 0.0
#         obs = env.reset()
