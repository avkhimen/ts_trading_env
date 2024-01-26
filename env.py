import pandas as pd
import numpy as np
import gymnasium
import random

class TSEnv():
    def __init__(self, ts, obs_dim, action_dim, seed, lookup_interval, period_interval):
        self.ts = np.array(ts)
        self.seed = seed
        self.observation_space = np.zeros((obs_dim, ))
        self.action_space = gymnasium.spaces.Discrete(action_dim, seed=self.seed)
        self.lookup_interval = lookup_interval
        self.period_interval = period_interval

    def reset(self):
        """Returns the state at the beginning of an episode.
           The environment must maintain state."""
        seed=self.seed
        print('--------------------')
        print(random.choice(range(self.lookup_interval)))
        print(len(self.ts) - self.period_interval)
        print('--------------------')
        ind = random.choice(range(self.lookup_interval,len(self.ts) - self.period_interval))
        price_0 = self.ts[ind - self.lookup_interval]
        self.step = 0
        state = [self.ts[ind - self.lookup_interval : ind + self.period_interval + 1] / price_0, price_0, 1, 0, step] #own cash at start
        # state = {past prices, past_volumes, price_0, past action, ownership status, volume_0, step}
        # state = [price1, price2, past action, ownership status]
        # the last price must be the price for time + 1
        # actions:
        # 0 - buy crypto
        # 1 - sell crypto
        # ownership status:
        # 0 - own cash
        # 1 - own crypto
        self.state = state
        return state
    
    def step(self, action):
        """Uses action to return next state, reward, done, and info.
           The environment must maintain state."""
        next_state = None
        done = False
        self.step += 1
        if self.step == self.period_interval:
            done = True
        reward = self.calculate_reward(next_state)
        info = {}
        self.state = next_state
        return next_state, reward, done, info

    def calculate_reward(self, next_state):
        """Calculates the reward"""
        return None

# Test
env = TSEnv([1,2,3,4,5,6,7,8,9], 3, 2, 32, 3, 3)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(env.reset())