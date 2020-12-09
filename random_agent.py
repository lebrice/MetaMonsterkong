import logging
import os, sys
import numpy as np
import gym
from gym.wrappers import Monitor
import gym_ple
from matplotlib import pyplot as plt
import cv2
import meta_monsterkong

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space, num_envs):
        self.action_space = action_space
        self.num_envs = num_envs

    def act(self, observation, reward, done):
        return np.array([self.action_space.sample() for _ in range(self.num_envs)])

if __name__ == '__main__':
    
    mk_config = {
        'MapsDir':'./meta_monsterkong/maps20x20',
        'MapHeightInTiles': 20,
        'MapWidthInTiles': 20,
        'IsRender':True,
        'SingleID': None,
        'DenseRewardCoeff':0.001,
        'RewardsWin':50.0,
        'StartLevel':0,
        'NumLevels':200,
        'TextureFixed':False,
        'Mode': 'Train',    
    }

    num_envs = 10
    envs = meta_monsterkong.make_vec_random_env(num_envs=num_envs, mk_config=mk_config)
    agent = RandomAgent(envs.action_space, num_envs)

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        ob = envs.reset()
        envs.render(mode='human')
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = envs.step(action)
            envs.render(mode='human')         
            if done[0]:
                break
        print(np.mean(reward))

    envs.close()
