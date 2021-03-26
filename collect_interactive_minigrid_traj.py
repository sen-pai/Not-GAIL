import os
import matplotlib.pyplot as plt
from imitation.data.types import Trajectory
import pickle5 as pickle

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

"""
After you are done, click ESC to write the data, dont just close the window. 
"""


traj_dataset = []
obs_list = []
action_list = []
info_list = []



def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    global obs_list, action_list, info_list
    obs_list = []
    action_list = []
    info_list = []


    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()
    obs_list.append(obs)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action, action_int):
    global obs_list, action_list, info_list, traj_dataset
    obs, reward, done, info = env.step(action)
    action_list.append(action_int)
    info_list.append(info)
    obs_list.append(obs)
    print(obs.shape)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        print(len(action_list), len(obs_list))
        traj_dataset.append(Trajectory(obs = np.array(obs_list), acts= np.array(action_list), infos = np.array(info_list)))
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left, 0)
        return
    if event.key == 'right':
        step(env.actions.right, 1)
        return
    if event.key == 'up':
        step(env.actions.forward, 2)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle, 5)
        return
    if event.key == 'pageup':
        step(env.actions.pickup, 3)
        return
    if event.key == 'pagedown':
        step(env.actions.drop, 4)
        return

    if event.key == 'enter':
        step(env.actions.done, 6)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-LavaCrossingS9N1-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)


args = parser.parse_args()

env = gym.make(args.env)


env = FlatObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)


with open('lava_ideal.pkl', 'wb') as handle:
    pickle.dump(traj_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote to pkl")
