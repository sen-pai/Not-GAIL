import gym
import gym_minigrid
from gym_minigrid import wrappers
from stable_baselines3 import PPO
from gym.wrappers.frame_stack import FrameStack

env = gym.make('MiniGrid-Empty-Random-6x6-v0')
env = wrappers.FullyObsWrapper(env)
env = wrappers.FlatObsWrapper(env)
# env = FrameStack(env, 4 )
# env = wrappers.RGBImgObsWrapper(env)
# env = wrappers.ImgObsWrapper(env)


model = PPO('MlpPolicy', env, verbose=1)
print(model.env.reset())
print(model.env.reset().shape)

model.learn(total_timesteps= int(1e5))

model.save("ppo_minigrid_empty_fully_obs")