import pandas as pd
import numpy as np

class TSEnv():
    def __init__(self, ts):
        self.ts = ts
        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n
        # self.env.step(action)
        # self.env.reset(seed=self.seed)