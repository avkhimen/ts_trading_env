import pandas as pd
import numpy as np
import gymnasium

class TSEnv():
    def __init__(self, ts, obs_dim, action_dim, seed):
        self.ts = ts
        self.seed = seed
        self.observation_space = np.zeros((obs_dim, ))
        self.action_space = gymnasium.spaces.Discrete(action_dim, seed=self.seed)

    def reset(self):
        """Returns the state at the beginning of an episode"""
        seed=self.seed
        state = None # state = {past prices, past_volumes, past action, ownership status}
        # state = [price1, price2, past action, ownership status]
        # the last price must be the price for time + 1
        # actions:
        # 0 - do nothing
        # 1 - buy crypto
        # 2 - sell crypto
        # ownership status:
        # 0 - own cash
        # 1 - own crypto
        return state
    
    def step(self, action):
        """Uses action to return next state, reward, done, and info"""
        next_state = None
        done = None
        reward = None
        info = {}
        return next_state, reward, done, info

    def calculate_reward(self):
        """Calculates the reward"""
        return None

# Test
env = TSEnv([1,2,3], 3, 2, 32)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())