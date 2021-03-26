import gym
import gym_minigrid
from gym_minigrid import wrappers
from stable_baselines3 import PPO


env = gym.make('MiniGrid-LavaGapS5-v0')
env = wrappers.FlatObsWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)
model.load("ppo_minigrid_empty_8")

model.learn(total_timesteps= int(2e4))

model.save("ppo_minigrid_curriculum_lava_5")