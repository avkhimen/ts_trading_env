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
        self.ind = random.choice(range(self.lookup_interval,len(self.ts) - self.period_interval))
        self.price_0 = self.ts[self.ind - self.lookup_interval]
        self.step_ = 0
        self.own_status = 0
        state = [self.ts[self.ind - self.lookup_interval : self.ind \
                 + self.period_interval + 1] / self.price_0, self.price_0, 1, self.own_status, self.step_] #own cash at start
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
        reward = self.calculate_reward(action)
        self.next_state = self.get_next_state(action)
        done = False
        if self.step_ == self.period_interval:
            done = True
        info = {}
        if not done:
            self.state = self.next_state
        return self.next_state, reward, done, info

    def get_next_state(self, action):
        self.ind += 1
        self.step_ += 1
        self.price_0 = self.ts[self.ind - self.lookup_interval]
        if action == 0:
            self.own_status = 1
        else:
            self.own_status = 0
        self.next_state = [self.ts[self.ind - self.lookup_interval : self.ind \
                           + self.period_interval + 1] / self.price_0, self.price_0, 1, self.own_status, self.step_] #own cash at start
        return self.next_state

    def calculate_reward(self, action):
        """Calculates the reward"""
        real_prices = self.state[0] * self.state[1]
        real_price_difference = real_prices[-1] - real_prices[-2]
        if self.own_status == 0:
            reward = -(real_price_difference)
        else:
            reward = real_price_difference
        return reward

# Test
env = TSEnv([1,2,3,4,5,6,7,8,9], 3, 2, 32, 3, 3)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(env.reset())
print(env.step(1))
print(env.step(0))