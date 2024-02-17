import pandas as pd
import numpy as np
import gymnasium
import random
import itertools

class TSEnv():
    # TODO: change actions to buy, sell, hold
    # TODO: correct state to include only stock prices that are known
    def __init__(self, ts, action_dim, lookup_interval, period_interval, seed):

        assert len(ts) > lookup_interval + period_interval, 'time series length should be greater than ' + \
                                                            'the sum of lookup interval and period interval' 
        self.ts = np.array(ts)
        self.seed = seed
        self.observation_space = np.zeros((lookup_interval + 3, ))
        self.action_space = gymnasium.spaces.Discrete(action_dim)
        self.lookup_interval = lookup_interval
        self.period_interval = period_interval

    def reset(self):
        """Returns the state at the beginning of an episode.
           The environment must maintain state."""
        seed=self.seed
        self.ind = random.choice(range(self.lookup_interval,len(self.ts) - self.period_interval))
        print('ind is', self.ind)
        self.price_0 = self.ts[self.ind - self.lookup_interval]
        print('price_0 is', self.price_0)
        self.step_ = 0
        self.own_status = 0
        state = (self.ts[self.ind - self.lookup_interval : self.ind] / self.price_0).tolist()
        state.extend([1, self.own_status, self.step_]) #own cash at start
        # state = {past prices, price_0, past action, ownership status, step}
        # the last price must be the price for time + 1
        # actions:
        # 0 - buy crypto
        # 1 - sell crypto
        # 2 - hold asset
        # ownership status:
        # 0 - own cash
        # 1 - own crypto
        state = np.array(state)
        info = {}
        return state, info
    
    def step(self, action):
        """Uses action to return next state, reward, done, truncated, and info.
           The environment must maintain state."""

        next_state = self.get_next_state(action)
        reward = self.calculate_reward(action, next_state)
        
        # calculate done
        done = False
        if self.step_ == self.period_interval:
            done = True

        # calculate info
        info = {}

        # calculate truncate
        truncated = False
        
        return next_state, reward, done, truncated, info

    def close(self):
        return None

    def get_next_state(self, action):

        # Increment ind
        self.ind += 1

        # Increment step_
        self.step_ += 1

        # actions:
        # 0 - buy crypto
        # 1 - sell crypto
        # 2 - hold asset
        # ownership status:
        # 0 - own cash
        # 1 - own crypto

        # Get new own_status
        if action == 0 and self.own_status == 0:
            own_status = 1
        elif action == 0 and self.own_status == 1:
            own_status = 1
        elif action == 1 and self.own_status == 0:
            own_status = 0
        elif action == 1 and self.own_status == 1:
            own_status = 0
        elif action == 2 and self.own_status == 0:
            own_status = 0
        elif action == 2 and self.own_status == 1:
            own_status = 1

        self.own_status = own_status

        next_state = (self.ts[self.ind - self.lookup_interval : self.ind] / self.price_0).tolist()
        next_state.extend([action, self.own_status, self.step_])
        next_state = np.array(next_state)

        return next_state

    def calculate_reward(self, action, next_state, cash_mult=1, crypto_mult=1):
        """Calculates the reward"""

        scaled_real_prices = np.array(next_state[:-3])
        print('step:', self.step_, 'scaled_real_prices', scaled_real_prices)
        scaled_price_difference = scaled_real_prices[-1] - scaled_real_prices[-2]

        # if own_status = cash = 0:
        # if price goes up -diff -> stays negative
        # if price down down -diff -> turns to positive
        # if own_status = crypto = 1:
        # if price goes up diff -> stays positive
        # if price does down diff -> stays negative

        if action == 2:
            reward = 0
        elif (self.own_status == 0) and (action != 2):
            reward = -cash_mult * scaled_price_difference
        elif (self.own_status == 1) and (action != 2):
            reward = crypto_mult * scaled_price_difference
        
        return reward

if __name__ == '__main__':
    # Test
    ts = [10,20,30,40,50,60,70]
    print(ts)
    env = TSEnv(ts=ts, action_dim=3, lookup_interval=3, period_interval=3, seed=32)

    def select_action(state):
        return env.action_space.sample()

    state, info = env.reset()
    print('state after reset is', state)
    reward_total = 0
    for _ in range(env.period_interval):
        action = select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        reward_total += reward
        state = next_state

    print(reward_total)