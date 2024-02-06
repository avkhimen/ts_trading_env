import pandas as pd
import numpy as np
import gymnasium
import random
import itertools

class TSEnv():
    def __init__(self, ts, action_dim, lookup_interval, period_interval, seed):

        assert len(ts) > lookup_interval + period_interval, 'time series length should be greater than \
        the sum of lookup interval and period interval' 
        self.ts = np.array(ts)
        self.seed = seed
        self.observation_space = np.zeros((lookup_interval + 4, ))
        self.action_space = gymnasium.spaces.Discrete(action_dim, seed=self.seed)
        self.lookup_interval = lookup_interval
        self.period_interval = period_interval

    def reset(self):
        """Returns the state at the beginning of an episode.
           The environment must maintain state."""
        seed=self.seed
        self.ind = random.choice(range(self.lookup_interval,len(self.ts) - self.period_interval))
        self.price_0 = self.ts[self.ind - self.lookup_interval]
        self.step_ = 0
        self.own_status = 0
        #print((self.ts[self.ind - self.lookup_interval : self.ind +   1] / self.price_0).tolist())
        state = (self.ts[self.ind - self.lookup_interval : self.ind +   1] / self.price_0).tolist()
        state.extend([self.price_0, 1, self.own_status, self.step_]) #own cash at start
        # state = {past prices, past_volumes, price_0, past action, ownership status, volume_0, step}
        # state = [price1, price2, past action, ownership status]
        # the last price must be the price for time + 1
        # actions:
        # 0 - buy crypto
        # 1 - sell crypto
        # ownership status:
        # 0 - own cash
        # 1 - own crypto
        self.state = np.array(state)
        info = {}
        return state, info
    
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

    def close(self):
        return None

    def get_next_state(self, action):
        self.ind += 1
        self.step_ += 1
        self.price_0 = self.ts[self.ind - self.lookup_interval]
        if action == 0:
            self.own_status = 1
        else:
            self.own_status = 0
        self.next_state = (self.ts[self.ind - self.lookup_interval : self.ind + 1] / self.price_0).tolist()
        self.next_state.extend([self.price_0, action, self.own_status, self.step_]) #own cash at start
        self.next_state = np.array(self.next_state)

        return self.next_state

    def calculate_reward(self, action):
        """Calculates the reward"""
        real_prices = np.array(self.state[:-4]) * self.state[-4]
        real_price_difference = real_prices[-1] - real_prices[-2]
        # if own_status = cash = 0:
        # if price goes up -diff -> stays negative
        # if price down down -diff -> turns to positive
        # if own_status = crypto = 1:
        # if price goes up diff -> stays positive
        # if price does down diff -> stays negative
        if self.own_status == 0:
            reward = -(real_price_difference)
        else:
            reward = real_price_difference
        return reward

if __name__ == '__main__':
    # Test
    env = TSEnv(ts=[*range(1,8)], action_dim=2, lookup_interval=3, period_interval=3, seed=32)
    print(env.reset())
    reward_total = 0
    for _ in range(1):
        next_state, reward, done, _ = env.step(env.action_space.sample())
        reward_total += reward

    print(reward_total)