import gym
import gym_minigrid
from gym_minigrid import wrappers
from stable_baselines3 import PPO


env = gym.make('MiniGrid-Empty-Random-6x6-v0')
env = wrappers.FlatObsWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps= int(5e4))

model.save("ppo_minigrid_empty")