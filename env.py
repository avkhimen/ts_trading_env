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
        seed=self.seed
        pass
    
    def step(self, action):
        pass

# Test
env = TSEnv([1,2,3], 3, 2, 32)
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())