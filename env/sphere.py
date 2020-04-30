from typing import Optional

import gym
import numpy as np


class Sphere(gym.Env):
    """Just a Sphere function for debugging purposes

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    NUM_DIMS: int = 10

    last_action: Optional[np.ndarray]
    last_reward: Optional[float]
    observation: np.ndarray

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.NUM_DIMS,), dtype=np.float32)

        self.observation = np.zeros(1,)
        self.reset()

    @property
    def is_done(self):
        return self.last_action is not None

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            self.reset()
            return seed

    def reset(self):
        self.last_action = None
        self.last_reward = None
        return self.observation

    def step(self, action: np.ndarray):
        self.last_action = action
        reward = -float(sum(np.square(action)))
        return self.observation, reward, self.is_done, {}

    def render(self, mode='human', close=False):
        print(f'observation: {self.observation},'
              f'\tlast_action: {self.last_action},'
              f'\treward: {self.last_reward},'
              f'\tdone: {self.is_done}')

