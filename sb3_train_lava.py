import gym
import gym_minigrid
from gym_minigrid import wrappers
from stable_baselines3 import PPO


env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
# env = wrappers.ActionBonus(env)
env = wrappers.StateBonus(env)

env = wrappers.FlatObsWrapper(env)

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps= int(3e4))

model.save("ppo_minigrid_lava_bad")